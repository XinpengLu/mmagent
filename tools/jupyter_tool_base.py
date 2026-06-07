"""Shared stateful Jupyter execution engine for Python-backed tools."""

import asyncio
import ast
import atexit
import base64
import concurrent.futures
import io
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Dict, Set

from PIL import Image
from jupyter_client import BlockingKernelClient
from tools.base import BaseTool

# Constants
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
KERNEL_INIT_POLL_INTERVAL = 0.1  # seconds
KERNEL_INIT_MAX_WAIT = 30  # seconds
KERNEL_INIT_SLEEP = 0.1  # seconds after initialization

# Global variables for kernel management
_KERNEL_CLIENTS: dict = {}
_MISC_SUBPROCESSES: Dict[str, subprocess.Popen] = {}
_KERNEL_LOCK = threading.Lock()  # Thread lock to protect global dictionaries

LAUNCH_KERNEL_PY = """
from ipykernel import kernelapp as app
app.launch_new_instance()
"""


def _kill_kernels_and_subprocesses(_sig_num=None, _frame=None):
    """Clean up all kernels and subprocesses."""
    with _KERNEL_LOCK:
        for v in list(_KERNEL_CLIENTS.values()):
            try:
                v.shutdown()
            except:
                pass
        _KERNEL_CLIENTS.clear()
        
        for v in list(_MISC_SUBPROCESSES.values()):
            try:
                v.terminate()
            except:
                pass
        _MISC_SUBPROCESSES.clear()


# Register cleanup function
if threading.current_thread() is threading.main_thread():
    atexit.register(_kill_kernels_and_subprocesses)


def execute_with_timeout(func, timeout, **kwargs):
    """Execute a function with timeout support and graceful interruption.
    
    Note: Python threads cannot be truly killed, so after timeout:
    1. Set stop_event to notify the function to exit
    2. Use wait=False to avoid blocking
    3. Caller must ensure kernel process is forcefully terminated
    """
    import time as time_module
    import threading
    start_time = time_module.time()
    log_name = kwargs.pop("_log_name", "JupyterTool")
    
    # Create stop event for graceful interruption
    stop_event = threading.Event()
    kwargs['_stop_event'] = stop_event
    
    # Don't use with statement to avoid waiting for task completion on exit
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(func, **kwargs)
        try:
            result = future.result(timeout=timeout)
            elapsed = time_module.time() - start_time
            print(f"[{log_name}] execute_with_timeout completed in {elapsed:.2f}s (timeout was {timeout}s)")
            return result
        except concurrent.futures.TimeoutError:
            # Set stop event to notify function to exit early
            stop_event.set()
            # Try to cancel task (may be ineffective for already running tasks, but at least marks as cancelled)
            future.cancel()
            elapsed = time_module.time() - start_time
            print(f"[{log_name}] Execution timeout after {elapsed:.2f}s (limit: {timeout}s), stop event set")
            return f'TimeoutError: Function execution timed out after {timeout} seconds.'
        except Exception as e:
            # Also set stop event when exception occurs
            stop_event.set()
            elapsed = time_module.time() - start_time
            print(f"[{log_name}] execute_with_timeout exception after {elapsed:.2f}s: {e}")
            return f'Fatal Error: {e}'
    finally:
        # Use wait=False to return immediately, don't wait for timed-out tasks to complete
        # Timed-out tasks will continue running in the background until completed or terminated by kernel
        executor.shutdown(wait=False)


class JupyterKernelToolBase(BaseTool):
    """Base class for tools that execute Python in a stateful Jupyter kernel."""

    name = "jupyter_tool"
    tool_log_name = "JupyterTool"
    output_image_prefix = "tool_image"
    default_base_dir = "/tmp/jupyter_tool"
    auto_capture_pil_variables = False
    description = '''
Executes Python code in a stateful Jupyter kernel and return results.

Capabilities:
- Use PIL, NumPy, or OpenCV to process or analyze images to improve understanding.
- Perform complex mathematical calculations or data manipulation.
- Code execution is persistent: variables and functions are stored and reusable in subsequent calls.

Image Variables (Pre-loaded as PIL Image objects):
- original_image / original_image_N (N=1,2,3,...): The original input images are already loaded as PIL Image objects. The first image is also available as image.
  The following code is automatically executed before your code:
  ```python
  from PIL import Image
  original_image = Image.open('...')
  original_image_1 = Image.open('...')  # if multiple images exist
  ```
  You can directly use these variables without loading them again.
- tool_image_N (N=1,2,3,...): Images generated by your previous code calls.
- observation_N (N=1,2,3,...): Images generated by zoom tool operations.

Visualization & Output:
- To view generated images, return the object, use plt.show(), or save it (e.g., image.save('out.png')). 
- DO NOT use image.show(). 
- Displayed images will be available as "tool_image_N" in the next turn.

Pre-installed packages:
- PIL
- NumPy
- OpenCV
- Matplotlib
- SciPy
- Scikit-learn
- Scikit-image
- Pandas
- SymPy
'''
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute in the Jupyter kernel"
            }
        },
        "required": ["code"]
    }
    
    def __init__(self, config=None):
        super().__init__(config)
        # Instance ID (generated before setting work directory)
        self.instance_id = str(uuid.uuid4())
        
        # Set up work directory: create independent directory for each instance to avoid file conflicts
        if config and 'work_dir' in config:
            # If work_dir is specified in config, use it as base directory and create subdirectory per instance
            # This ensures each instance has independent workspace to avoid conflicts during parallel execution
            base_dir = config['work_dir']
            self.work_dir = os.path.join(base_dir, self.instance_id)
        else:
            # Otherwise use default directory based on instance_id
            self.work_dir = os.path.join(self.default_base_dir, self.instance_id)
        
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Output detection timeout (to detect if kernel is stuck)
        # If no valid output (text/image/error) within this time, consider it timeout
        # This is the only timeout mechanism, simpler and stricter
        self.output_timeout = config.get('output_timeout', 30) if config else 30  # default 30 seconds
        # If set to None or 0, considered disabled (not recommended, may cause infinite wait)
        if self.output_timeout is None or self.output_timeout == 0:
            self.output_timeout = None
            # Raise exception immediately instead of waiting until execution
            raise ValueError("output_timeout must be set and > 0. It is required for timeout detection.")
    
    def _is_kernel_file(self, filename: str) -> bool:
        """Check if a file is kernel-related and should not be deleted."""
        return filename.startswith(('kernel_connection_file_', 'launch_kernel_'))
    
    def _is_image_file(self, filename: str) -> bool:
        """Check if a file is an image based on extension."""
        return filename.lower().endswith(IMAGE_EXTENSIONS)
    
    def _get_kernel_id(self) -> str:
        """Generate kernel ID for current instance, process and thread."""
        import threading
        thread_id = threading.get_ident()
        return f'{self.instance_id}_{os.getpid()}_{thread_id}'
    
    def _process_image_output(self, msg: dict, served_images: set, tool_image_counter: int) -> tuple:
        """Convert image output from one kernel message into Markdown."""
        data = msg.get('content', {}).get('data', {})
        if 'image/png' in data:
            image_b64 = data['image/png']
            image_url = self._serve_image(image_b64)
            if os.path.isfile(image_url):
                served_images.add(os.path.basename(image_url))
            tool_image_counter += 1
            return f'![{self.output_image_prefix}_{tool_image_counter}]({image_url})', tool_image_counter
        return '', tool_image_counter
    
    def _clean_work_dir(self, input_image_files: Set[str] = None):
        """
        Clean input image files in work directory while preserving user-saved images.
        
        Cleaning strategy:
        1. Preserve kernel-related files (kernel_connection_file_*, launch_kernel_*)
        2. Only delete input image files (passed via image_map, will be re-injected in this call)
        3. Preserve user-saved image files (for cross-step access)
        4. Delete other user-generated files (.py, .txt, .csv, etc., excluding kernel files)
        """
        if not os.path.exists(self.work_dir):
            return
        
        # Track input image filenames (passed via image_map)
        input_files = input_image_files if input_image_files else set()
        
        try:
            for f in os.listdir(self.work_dir):
                file_path = os.path.join(self.work_dir, f)
                
                # Skip kernel-related files
                if self._is_kernel_file(f):
                    continue
                
                # Only delete input image files, preserve user-saved images
                if self._is_image_file(f):
                    if f in input_files:
                        # This is an input image, can be deleted (will be re-injected in this call)
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
                    # else: user-saved image, keep it
                
                # Delete other user-generated files (but preserve kernel files)
                if f.endswith(('.py', '.txt', '.csv', '.json')) and not f.startswith('kernel_'):
                    if 'kernel' not in f.lower() and 'connection' not in f.lower():
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
        except Exception as e:
            # Cleaning failure should not affect code execution
                print(f"[{self.tool_log_name}] Warning: Failed to clean work_dir: {e}")
    
    def _code_from_params(self, params) -> str:
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                code = params
            else:
                code = params.get('code', '') if isinstance(params, dict) else str(params)
        else:
            code = params.get('code', '')

        triple_match = re.search(r'```[^\n]*\n(.+?)```', code, re.DOTALL)
        if triple_match:
            code = triple_match.group(1)
        return code

    def _input_image_files(self, image_map: Dict[str, Image.Image] = None) -> Set[str]:
        input_image_files = set()
        if image_map:
            for name in image_map.keys():
                var_name = re.sub(r'[^a-zA-Z0-9_]', '', name)
                if var_name:
                    for ext in IMAGE_EXTENSIONS:
                        input_image_files.add(f'{var_name}{ext}')
                if name == 'original_image' or name.startswith('original_image_'):
                    for ext in IMAGE_EXTENSIONS:
                        input_image_files.add(f'image{ext}')
        return input_image_files

    def _sorted_image_items(self, image_map: Dict[str, Image.Image]):
        return sorted(image_map.items(), key=lambda x: (
            0 if x[0] == 'original_image' else (1 if x[0].startswith('original_image_') else 2),
            x[0]
        ))

    def _image_load_code(self, image_map: Dict[str, Image.Image] = None) -> str:
        image_load_code = ''
        if image_map:
            image_load_code += 'from PIL import Image\n'
            for name, img in self._sorted_image_items(image_map):
                var_name = re.sub(r'[^a-zA-Z0-9_]', '', name)
                if not var_name:
                    continue
                file_path = os.path.join(self.work_dir, f'{var_name}.png')
                img.save(file_path)
                image_load_code += f"{var_name} = Image.open({file_path!r})\n"

            if 'original_image' in image_map:
                image_alias_path = os.path.join(self.work_dir, 'image.png')
                image_map['original_image'].save(image_alias_path)
                image_load_code += "image = original_image\n"
        return image_load_code

    def _validate_code(self, code: str) -> str:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f'SyntaxError: {e}'

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ('exit', 'quit'):
                    return 'Error: Using exit() is not allowed'
        return ""

    def _existing_image_files(self) -> Set[str]:
        existing_image_files = set()
        if os.path.exists(self.work_dir):
            for f in os.listdir(self.work_dir):
                if self._is_image_file(f) and not self._is_kernel_file(f):
                    existing_image_files.add(f)
        return existing_image_files

    def _get_or_start_kernel(self, kernel_id: str):
        with _KERNEL_LOCK:
            kc = _KERNEL_CLIENTS.get(kernel_id)
        if kc is not None:
            return kc

        try:
            new_kc, new_subproc = self._start_kernel(kernel_id)
        except Exception as e:
            print(f"[{self.tool_log_name}] ERROR - Failed to start kernel {kernel_id}: {e}")
            import traceback
            traceback.print_exc()
            raise

        with _KERNEL_LOCK:
            if kernel_id not in _KERNEL_CLIENTS:
                _KERNEL_CLIENTS[kernel_id] = new_kc
                _MISC_SUBPROCESSES[kernel_id] = new_subproc
                return new_kc
            try:
                new_kc.shutdown()
                new_subproc.terminate()
            except Exception:
                pass
            return _KERNEL_CLIENTS[kernel_id]

    def _run_code_with_kernel(self, full_code, existing_image_files, kernel_id, tool_image_count):
        if self.output_timeout is None or self.output_timeout <= 0:
            raise ValueError("output_timeout must be set and > 0. It is required for timeout detection.")
        return execute_with_timeout(
            self._execute_code,
            self.output_timeout * 2,
            kc=self._get_or_start_kernel(kernel_id),
            code=full_code + '\n\n',
            existing_image_files=existing_image_files,
            kernel_id=kernel_id,
            tool_image_start_count=tool_image_count,
            _log_name=self.tool_log_name,
        )

    def call(self, params, image_map: Dict[str, Image.Image] = None, **kwargs):
        """Execute Python code, optionally with PIL images preloaded as variables."""
        code = self._code_from_params(params)
        if not code.strip():
            return 'Error: No code provided'

        self._clean_work_dir(self._input_image_files(image_map))
        tool_image_count = len([k for k in (image_map or {}) if k.startswith(f'{self.output_image_prefix}_')])
        full_code = self._image_load_code(image_map) + code

        validation_error = self._validate_code(code)
        if validation_error:
            return validation_error

        existing_image_files = self._existing_image_files()
        print(f"[{self.tool_log_name}] Executing code:\n{full_code[:300]}...")

        try:
            result = self._run_code_with_kernel(
                full_code,
                existing_image_files,
                self._get_kernel_id(),
                tool_image_count,
            )
            print(f"[{self.tool_log_name}] Execution completed")
            return result if result.strip() else 'Finished execution.'
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}"
            print(f"[{self.tool_log_name}] {error_msg}")
            return error_msg

    def _start_kernel(self, kernel_id: str):
        """Start a Jupyter kernel"""
        connection_file = os.path.join(self.work_dir, f'kernel_connection_file_{kernel_id}.json')
        launch_kernel_script = os.path.join(self.work_dir, f'launch_kernel_{kernel_id}.py')
        self._write_kernel_launch_script(connection_file, launch_kernel_script)
        kernel_process = self._launch_kernel_process(connection_file, launch_kernel_script)
        self._wait_for_connection_file(connection_file, kernel_process)
        kc = self._create_kernel_client(connection_file, kernel_process)
        self._initialize_kernel_client(kc, kernel_process)
        return kc, kernel_process

    def _write_kernel_launch_script(self, connection_file, launch_kernel_script):
        for path in [connection_file, launch_kernel_script]:
            if os.path.exists(path):
                os.remove(path)
        with open(launch_kernel_script, 'w') as fout:
            fout.write(LAUNCH_KERNEL_PY)

    def _launch_kernel_process(self, connection_file, launch_kernel_script):
        try:
            return subprocess.Popen(
                [
                    sys.executable,
                    os.path.abspath(launch_kernel_script),
                    '--IPKernelApp.connection_file',
                    os.path.abspath(connection_file),
                    '--matplotlib=inline',
                    '--quiet',
                ],
                cwd=os.path.abspath(self.work_dir),
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[CodeInterpreter] ERROR - Failed to start kernel process: {e}")
            raise

    def _terminate_process_quietly(self, process):
        try:
            process.terminate()
        except Exception:
            pass

    def _wait_for_connection_file(self, connection_file, kernel_process):
        max_wait = KERNEL_INIT_MAX_WAIT
        wait_time = 0
        while not os.path.isfile(connection_file) and wait_time < max_wait:
            time.sleep(KERNEL_INIT_POLL_INTERVAL)
            wait_time += KERNEL_INIT_POLL_INTERVAL
        
        if not os.path.isfile(connection_file):
            error_msg = f"Kernel connection file not created after {max_wait}s"
            print(f"[CodeInterpreter] ERROR - {error_msg}")
            self._terminate_process_quietly(kernel_process)
            raise RuntimeError(error_msg)

        while wait_time < max_wait:
            try:
                with open(connection_file, 'r') as fp:
                    json.load(fp)
                break
            except (json.JSONDecodeError, IOError) as e:
                time.sleep(KERNEL_INIT_POLL_INTERVAL)
                wait_time += KERNEL_INIT_POLL_INTERVAL
                if wait_time >= max_wait:
                    error_msg = f"Connection file not readable after {max_wait}s: {e}"
                    print(f"[CodeInterpreter] ERROR - {error_msg}")
                    self._terminate_process_quietly(kernel_process)
                    raise RuntimeError(error_msg)

    def _create_kernel_client(self, connection_file, kernel_process):
        try:
            return BlockingKernelClient(connection_file=connection_file)
        except Exception as e:
            print(f"[CodeInterpreter] ERROR - Failed to create BlockingKernelClient: {e}")
            self._terminate_process_quietly(kernel_process)
            raise

    def _initialize_kernel_client(self, kc, kernel_process):
        # Jupyter kernels may run on worker threads.
        try:
            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        except Exception:
            pass  # Ignore event loop policy setup failure
        
        try:
            kc.load_connection_file()
            kc.start_channels()
            kc.wait_for_ready()
        except Exception as e:
            print(f"[CodeInterpreter] ERROR - Failed to initialize kernel client: {e}")
            import traceback
            traceback.print_exc()
            try:
                kc.shutdown()
            except:
                pass
            self._terminate_process_quietly(kernel_process)
            raise
    
    def _is_valid_output(self, msg_type: str, msg: dict) -> bool:
        """Return True when a kernel message should reset the output timer."""
        if msg_type == 'status':
            # Status messages don't count as valid output (just state updates)
            return False
        elif msg_type == 'execute_result':
            # Check if there's actual content
            data = msg.get('content', {}).get('data', {})
            if 'image/png' in data:
                return True  # Images count as valid output
            text = data.get('text/plain', '')
            return bool(text and text.strip())  # Only counts if there's text content
        elif msg_type == 'display_data':
            # Check if there's image or text
            data = msg.get('content', {}).get('data', {})
            if 'image/png' in data:
                return True
            text = data.get('text/plain', '')
            return bool(text and text.strip())
        elif msg_type == 'stream':
            # Stream output counts as valid output
            text = msg.get('content', {}).get('text', '')
            return bool(text and text.strip())
        elif msg_type == 'error':
            # Errors also count as valid output (at least we know something went wrong)
            return True
        return False
    
    def _interrupt_kernel(self, kernel_id, kc):
        """Interrupt kernel execution.
        
        Attempts multiple methods to interrupt kernel:
        1. Send interrupt message via shell channel (if supported)
        2. Send SIGINT signal to kernel process
        3. If SIGINT fails, send SIGKILL to forcefully terminate
        4. Clean up kernel references in global dictionaries
        """
        if kernel_id is None:
            print(f"[CodeInterpreter] Cannot interrupt kernel: kernel_id is None")
            return
        
        # Method 1: Try sending interrupt message via shell channel
        try:
            # BlockingKernelClient might not have interrupt() method, try via shell channel
            if hasattr(kc, 'interrupt'):
                kc.interrupt()
                print(f"[CodeInterpreter] Kernel interrupt via interrupt() method")
        except Exception as e:
            print(f"[CodeInterpreter] Failed to interrupt kernel via client: {e}")
        
        # Method 2: Send SIGINT signal to kernel process, if ineffective use SIGKILL
        kernel_process = None
        try:
            with _KERNEL_LOCK:
                if kernel_id in _MISC_SUBPROCESSES:
                    kernel_process = _MISC_SUBPROCESSES[kernel_id]
        except Exception as e:
            print(f"[CodeInterpreter] Failed to get kernel process: {e}")
        
        if kernel_process and kernel_process.poll() is None:  # Process still running
            try:
                # First send SIGINT
                kernel_process.send_signal(signal.SIGINT)
                print(f"[CodeInterpreter] Kernel interrupt via SIGINT signal sent to process")
                
                # Wait a moment to see if process responds
                time.sleep(1.0)
                
                if kernel_process.poll() is None:
                    # SIGINT ineffective, use SIGKILL to forcefully terminate
                    print(f"[CodeInterpreter] Kernel process not responding to SIGINT, sending SIGKILL")
                    kernel_process.kill()  # Equivalent to send_signal(signal.SIGKILL)
                    time.sleep(0.5)
                    
                    if kernel_process.poll() is None:
                        print(f"[CodeInterpreter] WARNING: Kernel process still running after SIGKILL")
                    else:
                        print(f"[CodeInterpreter] Kernel process terminated via SIGKILL")
                else:
                    print(f"[CodeInterpreter] Kernel process terminated via SIGINT")
                    
            except Exception as e:
                print(f"[CodeInterpreter] Failed to terminate kernel process: {e}")
        
        # Method 3: Clean kernel references in global dictionary (ensure kernel is recreated on next call)
        try:
            with _KERNEL_LOCK:
                if kernel_id in _KERNEL_CLIENTS:
                    try:
                        _KERNEL_CLIENTS[kernel_id].shutdown()
                    except:
                        pass
                    del _KERNEL_CLIENTS[kernel_id]
                    print(f"[CodeInterpreter] Kernel client removed from cache")
                if kernel_id in _MISC_SUBPROCESSES:
                    del _MISC_SUBPROCESSES[kernel_id]
                    print(f"[CodeInterpreter] Kernel process removed from cache")
        except Exception as e:
            print(f"[CodeInterpreter] Failed to cleanup kernel references: {e}")

    def _scan_existing_images(self) -> Set[str]:
        image_files = set()
        if os.path.exists(self.work_dir):
            for f in os.listdir(self.work_dir):
                if self._is_image_file(f) and not self._is_kernel_file(f):
                    image_files.add(f)
        return image_files

    def _next_kernel_output(self, kc, single_wait_timeout, served_images, tool_image_counter):
        msg = kc.get_iopub_msg(timeout=single_wait_timeout)
        msg_type = msg.get('msg_type', 'unknown')
        text = ''
        image = ''
        finished = False

        if msg_type == 'status':
            finished = msg.get('content', {}).get('execution_state', 'unknown') == 'idle'
        elif msg_type == 'execute_result':
            content_data = msg.get('content', {}).get('data', {})
            text = content_data.get('text/plain', '')
            image, tool_image_counter = self._process_image_output(msg, served_images, tool_image_counter)
        elif msg_type == 'display_data':
            image, tool_image_counter = self._process_image_output(msg, served_images, tool_image_counter)
            if not image:
                content_data = msg.get('content', {}).get('data', {})
                text = content_data.get('text/plain', '')
        elif msg_type == 'stream':
            msg_type = msg.get('content', {}).get('name', 'stdout')
            text = msg.get('content', {}).get('text', '')
        elif msg_type == 'error':
            traceback_list = msg.get('content', {}).get('traceback', [])
            text = _escape_ansi('\n'.join(traceback_list))

        return msg, msg_type, text, image, finished, tool_image_counter

    def _append_new_image_files(self, result, existing_image_files, served_images, tool_image_counter):
        new_image_files = []
        if os.path.exists(self.work_dir):
            for f in os.listdir(self.work_dir):
                if self._is_image_file(f) and not self._is_kernel_file(f):
                    if f not in served_images and f not in existing_image_files:
                        new_image_files.append(f)

        for img_file in sorted(new_image_files):
            image_path = os.path.join(self.work_dir, img_file)
            if os.path.exists(image_path):
                tool_image_counter += 1
                abs_image_path = os.path.abspath(image_path)
                result += f'\n\n![{self.output_image_prefix}_{tool_image_counter}]({abs_image_path})'
                print(f"[{self.tool_log_name}] Auto-detected saved image: {img_file} -> {abs_image_path}")
        return result

    def _stop_or_timeout_text(self, kc, kernel_id, stop_event, last_output_time):
        if stop_event and stop_event.is_set():
            print(f"[CodeInterpreter] Stop event detected, interrupting execution")
            self._interrupt_kernel(kernel_id, kc)
            return 'TimeoutError: Execution was interrupted due to timeout.'

        time_since_last_output = time.time() - last_output_time
        if time_since_last_output < self.output_timeout:
            return ""

        print(f"[CodeInterpreter] Output timeout detected: {time_since_last_output:.1f}s without valid output")
        self._interrupt_kernel(kernel_id, kc)
        return (
            f'TimeoutError: No output received for {time_since_last_output:.1f} seconds '
            f'(output_timeout: {self.output_timeout}s). The kernel may be unresponsive or stuck.'
        )

    def _append_kernel_message_result(self, result, msg_type, text, image):
        if text:
            result += f'\n\n{msg_type}:\n\n```\n{text}\n```'
        if image:
            result += f'\n\n{image}'
        return result

    def _start_code_execution(self, kc, code: str) -> str:
        try:
            kc.wait_for_ready()
        except Exception as e:
            return f'Fatal Error: Kernel is not ready. {str(e)}'

        try:
            kc.execute(code)
        except Exception as e:
            return f'Fatal Error: Failed to execute code. {str(e)}'
        return ""

    def _poll_kernel_output(self, kc, served_images, tool_image_counter, last_output_time, stop_event):
        time_since_last_output = time.time() - last_output_time
        single_wait_timeout = min(max(1, self.output_timeout - time_since_last_output), 10)
        if stop_event and stop_event.is_set():
            raise queue.Empty

        msg, msg_type, text, image, finished, tool_image_counter = self._next_kernel_output(
            kc,
            single_wait_timeout,
            served_images,
            tool_image_counter,
        )
        if self._is_valid_output(msg_type, msg):
            last_output_time = time.time()
            print(f"[CodeInterpreter] Received valid output (type: {msg_type}), resetting output timeout timer")
        return msg_type, text, image, finished, tool_image_counter, last_output_time

    def _finish_code_execution(self, kc, code, result, existing_image_files, served_images, tool_image_counter):
        if result and "No output received" in result:
            print(f"[CodeInterpreter] Skipping image scan due to output timeout, returning immediately")
            return result.lstrip('\n')

        if self.auto_capture_pil_variables:
            self._capture_pil_variables(kc, code)

        return self._append_new_image_files(
            result,
            existing_image_files,
            served_images,
            tool_image_counter,
        ).lstrip('\n')
    
    def _execute_code(self, kc, code: str, existing_image_files=None, _stop_event=None, kernel_id=None, tool_image_start_count=0) -> str:
        """Execute code and collect text/image outputs from the kernel."""
        if existing_image_files is None:
            existing_image_files = self._scan_existing_images()
        
        served_images = set()
        start_error = self._start_code_execution(kc, code)
        if start_error:
            return start_error
        
        result = ''
        tool_image_counter = tool_image_start_count
        start_time = time.time()
        
        last_output_time = start_time
        output_timeout_enabled = self.output_timeout is not None and self.output_timeout > 0
        
        if not output_timeout_enabled:
            raise ValueError("output_timeout must be set and > 0. It is required for timeout detection.")
        
        while True:
            finished = False
            msg_type = 'error'
            
            timeout_text = self._stop_or_timeout_text(kc, kernel_id, _stop_event, last_output_time)
            if timeout_text:
                result = self._append_kernel_message_result(result, "error", timeout_text, "")
                break

            try:
                msg_type, text, image, finished, tool_image_counter, last_output_time = self._poll_kernel_output(
                    kc,
                    served_images,
                    tool_image_counter,
                    last_output_time,
                    _stop_event,
                )
            except queue.Empty:
                continue
                
            except (ValueError, ConnectionError, OSError) as e:
                error_msg = str(e)
                if result.strip():
                    finished = True
                else:
                    text = f'Error: {error_msg}. Kernel communication may be interrupted.'
                    finished = True
                    
            except BaseException as e:
                error_msg = str(e)
                error_type = type(e).__name__
                print(f"[CodeInterpreter] FATAL - {error_type}: {error_msg}")
                import traceback
                traceback.print_exc()
                
                text = f'Fatal Error ({error_type}): {error_msg}. The code interpreter encountered an unexpected error.'
                finished = True
            
            result = self._append_kernel_message_result(result, msg_type, text, image)
            if finished:
                break
        
        return self._finish_code_execution(
            kc,
            code,
            result,
            existing_image_files,
            served_images,
            tool_image_counter,
        )

    def _capture_pil_variables(self, kc, code: str = "") -> None:
        """Save user-created PIL images from the kernel so visual validation artifacts are visible."""
        prefix = uuid.uuid4().hex[:8]
        work_dir = os.path.abspath(self.work_dir)
        assigned_names = sorted(set(re.findall(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", code or "")))
        max_auto_capture = int(os.environ.get("CODE_INTERPRETER_MAX_AUTO_CAPTURE_IMAGES", "3") or 3)
        post_code = f"""import json as _xskill_json
import os as _xskill_os
import re as _xskill_re
try:
    from PIL import Image as _xskill_pil_image
    _xskill_dir = {work_dir!r}
    _xskill_prefix = {prefix!r}
    _xskill_assigned = {assigned_names!r}
    _xskill_max = {max_auto_capture!r}
    _xskill_excluded = {{'image', 'original_image'}}
    _xskill_preferred = ('overlay', 'bbox', 'box', 'mask', 'crop', 'annot', 'plot', 'visual', 'vis', 'result', 'processed', 'binary', 'binarized', 'threshold', 'rotated', 'enlarged', 'region')
    _xskill_saved = []
    _xskill_candidates = []
    for _xskill_name, _xskill_value in list(globals().items()):
        if not isinstance(_xskill_name, str):
            continue
        if _xskill_name.startswith('_') or _xskill_name in _xskill_excluded:
            continue
        if _xskill_name.startswith(('original_image', 'tool_image_', 'observation_')):
            continue
        if isinstance(_xskill_value, _xskill_pil_image.Image):
            if _xskill_assigned and _xskill_name not in _xskill_assigned:
                continue
            _xskill_score = 0
            _xskill_lower = _xskill_name.lower()
            if any(_term in _xskill_lower for _term in _xskill_preferred):
                _xskill_score += 2
            if _xskill_name in _xskill_assigned:
                _xskill_score += 1
            _xskill_candidates.append((_xskill_score, _xskill_name, _xskill_value))
    _xskill_candidates.sort(key=lambda _item: (-_item[0], _item[1]))
    for _, _xskill_name, _xskill_value in _xskill_candidates[:max(0, _xskill_max)]:
        _xskill_safe = _xskill_re.sub(r'[^a-zA-Z0-9_]+', '_', _xskill_name).strip('_') or 'image'
        _xskill_path = _xskill_os.path.join(_xskill_dir, f'auto_capture_{{_xskill_prefix}}_{{_xskill_safe}}.png')
        _xskill_value.save(_xskill_path)
        _xskill_saved.append(_xskill_path)
    if _xskill_saved:
        print('__XSKILL_CAPTURED_IMAGES__=' + _xskill_json.dumps(_xskill_saved))
except Exception as _xskill_exc:
    print('__XSKILL_CAPTURE_ERROR__=' + repr(_xskill_exc))
"""
        try:
            kc.execute(post_code)
            start = time.time()
            while time.time() - start < 5:
                try:
                    msg = kc.get_iopub_msg(timeout=1)
                except queue.Empty:
                    continue
                if msg.get('msg_type') == 'status' and msg.get('content', {}).get('execution_state') == 'idle':
                    break
        except Exception as e:
            print(f"[CodeInterpreter] Warning: failed to auto-capture PIL variables: {e}")
    
    def _serve_image(self, image_base64: str) -> str:
        """Save image and return path"""
        image_file = f'{uuid.uuid4()}.png'
        local_image_file = os.path.join(self.work_dir, image_file)
        
        png_bytes = base64.b64decode(image_base64)
        bytes_io = io.BytesIO(png_bytes)
        Image.open(bytes_io).save(local_image_file, 'png')
        
        return local_image_file
    
    def close(self):
        """Clean up the kernel and subprocess owned by this tool instance."""
        try:
            kernel_id = self._get_kernel_id()
            
            with _KERNEL_LOCK:
                if kernel_id in _KERNEL_CLIENTS:
                    try:
                        _KERNEL_CLIENTS[kernel_id].shutdown()
                    except:
                        pass
                    del _KERNEL_CLIENTS[kernel_id]
                if kernel_id in _MISC_SUBPROCESSES:
                    try:
                        _MISC_SUBPROCESSES[kernel_id].terminate()
                    except:
                        pass
                    del _MISC_SUBPROCESSES[kernel_id]
        except:
            # Ignore all errors to avoid exceptions during destruction
            pass

    def __del__(self):
        """Best-effort cleanup for resources that were not explicitly closed."""
        self.close()


def _escape_ansi(line: str) -> str:
    """Remove ANSI escape sequences"""
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)


# Event loop policy for kernels launched from worker threads.
if sys.platform == 'win32' and hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    _BasePolicy = asyncio.WindowsSelectorEventLoopPolicy
else:
    _BasePolicy = asyncio.DefaultEventLoopPolicy


class AnyThreadEventLoopPolicy(_BasePolicy):
    """Event loop policy that allows creating event loops in any thread."""
    
    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        try:
            return super().get_event_loop()
        except RuntimeError:
            loop = self.new_event_loop()
            self.set_event_loop(loop)
            return loop

