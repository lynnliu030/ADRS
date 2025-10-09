import os
import time
from bq.utils import g_str, r_str, y_str, b_str, get_distribution_info
import subprocess
import torch
import threading
import numpy as np

# EVOLVE-BLOCK-START
# The code is written to a file and then imported in ptq.py.
# Therefore, the code must be written in a string.
adaptive_bitrate_model_code = """
import torch

def adaptive_bitrate_model(inputs, weights, hessian_score, magnitude_score, layer_idx):
    \"\"\"
    This function is called for each layer in an LLM to adaptively quantize its weight matrix.
    The input to the function is the calibration input tensor and the weight matrix of the layer.
    Using the data from the calibration input and the weight matrix, it finds the optimal high-bitrate ratio and the columns to assign higher bitrate when quantized.'
    The quantization algorithm takes the num_columns * high_bitrate_ratio columns with the highest hybrid score and assigns them higher bitrate.
    \"\"\"
    
    def get_tensor_info(tensor):
        \"\"\"Get distribution info of the tensor.\"\"\"
        tensor_flat = tensor.flatten().to(torch.float64)
        if tensor_flat.shape[0] > 2**24:
            sample_idx = torch.randint(0, tensor_flat.shape[0], (2**24,))
            tensor_flat = tensor_flat[sample_idx]
        num = tensor_flat.size(0)
        avg = torch.mean(tensor_flat)
        min_val = torch.min(tensor_flat)
        max_val = torch.max(tensor_flat)
        std = torch.std(tensor_flat)
        skew = torch.mean((tensor_flat - avg) ** 3) / (std ** 3)
        kurtosis = torch.mean((tensor_flat - avg) ** 4) / (std ** 4)
        per_001 = torch.quantile(tensor_flat, 0.001)
        per_01 = torch.quantile(tensor_flat, 0.01)
        per_25 = torch.quantile(tensor_flat, 0.25)
        per_50 = torch.quantile(tensor_flat, 0.50)
        per_75 = torch.quantile(tensor_flat, 0.75)
        per_99 = torch.quantile(tensor_flat, 0.99)
        per_999 = torch.quantile(tensor_flat, 0.999)
        return {"num": num, avg: avg,
            "min": min_val, "max": max_val,
            "std": std.item(), "skew": skew.item(),
            "kurtosis": kurtosis, "per_001": per_001,
            "per_01": per_01, "per_25": per_25,
            "per_50": per_50, "per_75": per_75,
            "per_99": per_99, "per_999": per_999}
    w_info = get_tensor_info(weights)
    w_kurtosis = w_info['kurtosis']
    i_info = get_tensor_info(inputs)
    i_kurtosis = i_info['kurtosis']
    if torch.is_tensor(w_kurtosis):
        w_kurtosis = w_kurtosis.item()
    if torch.is_tensor(i_kurtosis):
        i_kurtosis = i_kurtosis.item()

    if w_kurtosis > 12 or i_kurtosis > 12:
        high_bitrate_ratio = 0.08
        alpha = 0.5
    elif w_kurtosis > 8 or i_kurtosis > 8:
        high_bitrate_ratio = 0.05
        alpha = 0.7
    elif w_kurtosis > 6 or i_kurtosis > 6: 
        high_bitrate_ratio = 0.08
        alpha = 0.7
    elif w_kurtosis > 5:
        high_bitrate_ratio = 0.10
        alpha = 0.75
    elif w_kurtosis > 4:
        high_bitrate_ratio = 0.25
        alpha = 0.75
    elif w_kurtosis > 3.5:
        high_bitrate_ratio = 0.35
        alpha = 0.8
    elif w_kurtosis > 3.05:
        high_bitrate_ratio = 0.52
        alpha = 0.9
    else:
        high_bitrate_ratio = 0.12
        alpha = 1.0
        
        
    hybrid_score = alpha * hessian_score + (1 - alpha) * magnitude_score
    hybrid_rank = torch.argsort(hybrid_score, descending=True)

    return high_bitrate_ratio, hybrid_rank
"""
# EVOLVE-BLOCK-END

def run_compression():
    model_name = "meta-llama/Meta-Llama-3-8B"
    rotation = "8B_R.bin"
    # model_name = "meta-llama/Llama-3.2-1B"
    # rotation = "1B_R.bin"
    method = "bq"
    key_bits = 16
    V = 8
    do_gptq = True
    
    adaptive_bitrate_model_path = "adaptive_bitrate_model.py"
    with open(adaptive_bitrate_model_path, "w") as f:
        f.write(adaptive_bitrate_model_code)
    
    def print_log(message):
        print(message) # Print to console with colors

    def stream_reader_thread(stream, stream_name_for_log, output_list,
                            print_log_func, color_func=None):
        try:
            for line in iter(stream.readline, ''): # Read until pipe closes
                if not line: # End of stream
                    break
                line = line.rstrip()
                log_prefix = f"[{stream_name_for_log}] "
                if color_func:
                    print_log_func(color_func(log_prefix) + line)
                else:
                    # For stdout, child's output might already be colored
                    print_log_func(line)
                output_list.append(line)
        except ValueError:
            # Can happen if pipe is closed abruptly
            print_log_func(r_str(f"[{stream_name_for_log}] Pipe closed or "
                                f"encoding error."))
        except Exception as e:
            print_log_func(
                r_str(f"[{stream_name_for_log}] Error reading stream: {e}")
            )
        
    iter_start_time = time.time()
    current_time_str = time.strftime("%H-%M-%S", time.localtime())
    num_gpus = torch.cuda.device_count()

    log_msg_header = b_str(
        f"[{current_time_str}] Running: "
    )
    log_msg_details = y_str(
        f"{model_name}, {method}, k={key_bits}, V={V}"
    )
    print_log(log_msg_header + log_msg_details)

    current_port = 25000 + (os.getpid() % 1000) * 20

    cmd = \
        f"torchrun --nnodes=1 --nproc_per_node={num_gpus} " + \
        f"--master_port={current_port} " + \
        f"ptq.py --input_model {model_name} " + \
        f"--do_train False --do_eval True " + \
        f"--per_device_eval_batch_size 4 " + \
        f"--model_max_length 2048 " + \
        f"--save_safetensors False " + \
        f"--w_bits {key_bits} " + \
        f"--w_clip " + \
        f"--w_groupsize {V} " + \
        f"--rotate " + \
        f"--save_qmodel_path {model_name}_{method}_{key_bits}_{V}_{do_gptq} " + \
        f"--optimized_rotation_path {rotation} "
    if not do_gptq:
        cmd += "--no_gptq "
    if method != "dummy":
        cmd += f"--use_{method} "

    print_log(g_str(f"Command:"))
    print_log(cmd)

    current_process = None
    # Initialize results for this iteration
    bitrate = "N/A"
    wikitext_ppl, ptb_ppl, c4_ppl = "N/A", "N/A", "N/A"
    c_qa, arc_c, arc_e = "N/A", "N/A", "N/A"
    hs, piqa, winogrande, avg_zs = "N/A", "N/A", "N/A", "N/A"
    time_taken_str = "N/A"
    status_for_csv = "STARTED"

    stdout_lines_list = []
    stderr_lines_list = []
    stdout_thread = None
    stderr_thread = None

    try:
        child_pid_str = (
            f"(PID: {current_process.pid if current_process else 'N/A'})"
        )
        print_log(g_str(f"--- Child Process Output {child_pid_str} ---"))

        current_process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        # Update PID string after Popen
        child_pid_str = f"(PID: {current_process.pid})"
        print_log(g_str(f"--- Child Process Output {child_pid_str} ---"))


        if current_process.stdout:
            stdout_thread = threading.Thread(
                target=stream_reader_thread,
                args=(current_process.stdout, "stdout",
                    stdout_lines_list, print_log)
            )
            stdout_thread.start()

        if current_process.stderr:
            stderr_thread = threading.Thread(
                target=stream_reader_thread,
                args=(current_process.stderr, "stderr",
                    stderr_lines_list, print_log, r_str)
            )
            stderr_thread.start()

        if stdout_thread: stdout_thread.join()
        if stderr_thread: stderr_thread.join()

        current_process.wait() # Wait for process to terminate
        return_code = current_process.returncode
        log_msg = (f"--- Child Process {child_pid_str} Finished "
                f"(Return Code: {return_code}) ---")
        print_log(g_str(log_msg))

        if return_code != 0:
            err_msg = (f"MAIN: Child for {model_name} exited with "
                    f"error code {return_code}.")
            print_log(r_str(err_msg))
            
        for line in stdout_lines_list:
            if "Average bitrate: " in line:
                bitrate = line.split("Average bitrate: ")[-1].strip()
            if "wikitext: " in line:
                wikitext_ppl = line.split("wikitext: ")[-1].strip()
            if "ptb: " in line:
                ptb_ppl = line.split("ptb: ")[-1].strip()

    except Exception as e:
        err_msg = (f"Error during subprocess execution or parsing "
                f"for {model_name}:")
        print_log(r_str(err_msg))
        print_log(r_str(f"Exception type: {type(e).__name__}, Msg: {e}"))
        import traceback
        detailed_error_info = traceback.format_exc()
        print_log(r_str("MAIN SCRIPT TRACEBACK FOR ITERATION ERROR:"))
        print_log(detailed_error_info)
        # Cleanup threads and process if they exist
        if stdout_thread and stdout_thread.is_alive():
            stdout_thread.join(timeout=2)
        if stderr_thread and stderr_thread.is_alive():
            stderr_thread.join(timeout=2)
        if current_process and current_process.poll() is None:
            current_process.kill()
            current_process.wait()

    finally: # Per-iteration finally block
        iter_end_time = time.time()
        time_taken_seconds = iter_end_time - iter_start_time
        time_taken_str = f"{time_taken_seconds:.2f}s"
        print_log(f"Iteration time: {time_taken_str}")
        
        output_dict = {
            "model_name": model_name,
            "method": method,
            "key_bits": key_bits,
            "V": V,
            "bitrate": bitrate,
            "do_gptq": do_gptq,
            "time_taken_str": time_taken_str,
            "wikitext_ppl": wikitext_ppl,
            "ptb_ppl": ptb_ppl,
        }
    
    return output_dict
    
if __name__ == "__main__":
    run_compression()