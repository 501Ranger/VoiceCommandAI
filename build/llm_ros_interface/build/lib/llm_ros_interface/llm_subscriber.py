import rclpy
from rclpy.node import Node
from std_msgs.msg import String # 导入 String 消息类型
import subprocess
import os
import threading # 用于在单独线程中读取进程输出
import queue # 用于线程间安全地传递数据

class LlmSubscriber(Node):

    def __init__(self):
        super().__init__('llm_subscriber')
        # 订阅 ASR 话题 - 保持不变
        self.subscription = self.create_subscription(
            String,
            '/audio_asr',
            self.listener_callback,
            10)
        self.subscription
        self.get_logger().info('LLM Subscriber Node has started.')

        # 创建一个发布器，用于发布 LLM 的回复
        self.publisher_ = self.create_publisher(String, '/llm_response', 10)
        self.get_logger().info('LLM Response Publisher created on topic /llm_response.')

        self.llama_cli_path = "/home/sunrise/llm_ws/llama.cpp/build/bin/llama-cli"
        self.model_path = "/home/sunrise/llm_ws/llama.cpp/qwen2.5-1.5b-instruct-q8_0.gguf"

        self.llama_process = None
        self.output_queue = queue.Queue() # 用于收集llama-cli的完整输出
        self.stop_event = threading.Event() # 用于通知读取线程停止
        self.response_received_event = threading.Event() # 用于标记一个完整的LLM回复已生成
        self.is_first_response_processed = False # 标记是否已处理并发布了当前 ASR 输入的第一个回复
        self.has_sent_initial_prompt = False # 新增：标记是否已发送初始化指令

        # 在节点启动时，一次性启动 llama-cli 进程
        self.start_llama_cli_persistent()

        # 创建一个定时器，定期检查并处理llama-cli的输出队列
        self.response_timer = self.create_timer(0.1, self.process_llm_output_queue)


    def start_llama_cli_persistent(self):
        """启动 llama-cli 进程并使其持久运行，并在启动成功后发送初始化指令"""
        llama_command = [
            self.llama_cli_path,
            "-m", self.model_path,
            "-n", "256", # 最大生成token数
            "--threads", "4", # 使用4个线程
            "-i", # 启用交互模式，使其持续读取stdin并写入stdout
            # "--keep", "-1", "-p", "<|im_start|>system\n你是一个有用的助手。<|im_end|>\n" # 可选：设置系统提示和保持上下文
        ]

        self.get_logger().info(f"正在启动 llama-cli 进程并使其持久运行: {' '.join(llama_command)}")
        try:
            self.llama_process = subprocess.Popen(
                llama_command,
                stdin=subprocess.PIPE,      # 捕获标准输入
                stdout=subprocess.PIPE,     # 捕获标准输出
                stderr=subprocess.PIPE,     # 捕获标准错误
                text=True,                  # 将输入/输出作为文本处理
                bufsize=1,                  # 行缓冲，重要，确保及时刷新
                universal_newlines=True     # 等同于 text=True
            )
            self.get_logger().info(f"llama-cli 进程已启动，PID: {self.llama_process.pid}")

            self.stdout_reader_thread = threading.Thread(
                target=self._read_llama_stdout,
                daemon=True
            )
            self.stdout_reader_thread.start()

            self.stderr_reader_thread = threading.Thread(
                target=self._read_llama_stderr,
                daemon=True
            )
            self.stderr_reader_thread.start()

            # 在进程启动后，如果尚未发送过初始化指令，则发送
            if not self.has_sent_initial_prompt:
                initial_prompt = "你是一个机器人动作指令解析器。你的任务是将用户的自然语言指令转换为标准化的机器人动作控制命令。请严格按照以下格式输出：是前进就回复0x01，后退就回复0x02,左转回复0x03，右转回复0x04，你今后所有的回复都只得按照此模板恢复，不要包含任何其他文字或解释，只输出一行符合格式的命令。"
                # 按照Qwen的聊天模板格式
                prompt_content_for_interactive = f"<|im_start|>user\n{initial_prompt}<|im_end|>\n<|im_start|>assistant\n"
                if not prompt_content_for_interactive.endswith('\n'):
                    prompt_content_for_interactive += '\n'
                
                self.llama_process.stdin.write(prompt_content_for_interactive)
                self.llama_process.stdin.flush()
                self.get_logger().info("已发送初始化指令给 llama-cli。")
                self.has_sent_initial_prompt = True # 标记已发送

        except FileNotFoundError:
            self.get_logger().error(f"找不到 llama-cli 可执行文件或模型文件。请检查路径: {self.llama_cli_path}, {self.model_path}")
            self.llama_process = None
        except Exception as e:
            self.get_logger().error(f"启动 llama-cli 进程时发生错误: {e}")
            self.llama_process = None

    def _read_llama_stdout(self):
        """单独线程函数：持续读取 llama-cli 的标准输出并放入队列"""
        if not self.llama_process or not self.llama_process.stdout:
            return

        self.get_logger().info("启动 llama-cli 标准输出读取线程。")
        current_response_lines = []
        try:
            for line in iter(self.llama_process.stdout.readline, ''):
                if self.stop_event.is_set():
                    break

                cleaned_line = line.strip()

                # 过滤掉已知的 llama.cpp 内部日志和与回复无关的行
                if cleaned_line.startswith("llama_perf_") or \
                   cleaned_line.startswith("build:") or \
                   cleaned_line.startswith("main:") or \
                   cleaned_line.startswith("gguf_init_from_file:") or \
                   cleaned_line.startswith("llama_model_load:") or \
                   cleaned_line.startswith("common_init_from_params:") or \
                   cleaned_line.startswith("Microseconds:") or \
                   cleaned_line.startswith("Prompt:") or \
                   cleaned_line.startswith("system_info:") or \
                   cleaned_line.startswith("sampler seed:") or \
                   cleaned_line.startswith("sampler params:") or \
                   cleaned_line.startswith("sampler chain:") or \
                   cleaned_line.startswith("generate:") or \
                   cleaned_line.startswith("== Running in interactive mode. ==") or \
                   cleaned_line.startswith("- Press Ctrl+C") or \
                   cleaned_line.startswith("- Press Return") or \
                   cleaned_line.startswith("- To return control") or \
                   cleaned_line.startswith("- If you want to submit") or \
                   cleaned_line.startswith("- Not using system message.") or \
                   cleaned_line.startswith("EOF by user"):
                    continue # 跳过这些内部消息

                # 如果行以 '>' 开头，表示 llama-cli 完成了一个回复段落
                if cleaned_line.startswith('>'):
                    # 提取 '>' 之后的内容作为当前回复的一部分
                    potential_response_part = cleaned_line[1:].strip()
                    if potential_response_part:
                            current_response_lines.append(potential_response_part)

                    # 如果当前有收集到的回复行，则组合并放入队列
                    if current_response_lines:
                        full_response = "\n".join(current_response_lines).strip()
                        # 清除所有聊天模板标记和多余的换行符
                        full_response = full_response.replace("<|im_start|>user", "").replace("<|im_end|>", "").replace("<|im_start|>assistant", "").strip()
                        full_response = full_response.replace("\n\n", "\n") # 移除双重换行，使输出更整洁

                        if full_response: # 确保回复非空才放入队列
                            self.output_queue.put(full_response)
                            self.response_received_event.set() # 标记一个完整回复已就绪
                        current_response_lines = [] # 重置，准备收集下一个回复
                    continue # 跳过包含 '>' 的行本身

                # 如果不是过滤掉的内部行，也不是以 '>' 开头的行，则认为是回复内容的一部分
                if cleaned_line:
                    current_response_lines.append(cleaned_line)

        except ValueError:
            self.get_logger().info("llama-cli 标准输出管道已关闭。")
        except Exception as e:
            self.get_logger().error(f"读取 llama-cli 标准输出时发生错误: {e}")
        finally:
            # 确保在进程意外退出时，任何剩余的缓冲行也能被处理
            if current_response_lines:
                full_response = "\n".join(current_response_lines).strip()
                full_response = full_response.replace("<|im_start|>user", "").replace("<|im_end|>", "").replace("<|im_start|>assistant", "").strip()
                full_response = full_response.replace("\n\n", "\n")
                if full_response:
                    self.output_queue.put(full_response)
                    self.response_received_event.set()
            self.get_logger().info("llama-cli 标准输出读取线程已停止。")


    def _read_llama_stderr(self):
        """单独线程函数：持续读取 llama-cli 的标准错误（用于日志和调试）"""
        if not self.llama_process or not self.llama_process.stderr:
            return

        self.get_logger().info("启动 llama-cli 标准错误读取线程。")
        try:
            for line in iter(self.llama_process.stderr.readline, ''):
                if self.stop_event.is_set():
                    break
                cleaned_line = line.strip()
                if cleaned_line and not cleaned_line.startswith("[INFO]"):
                    self.get_logger().warn(f"LLAMA_CLI_STDERR: {cleaned_line}")
        except ValueError:
            self.get_logger().info("llama-cli 标准错误管道已关闭。")
        except Exception as e:
            self.get_logger().error(f"读取 llama-cli 标准错误时发生错误: {e}")
        self.get_logger().info("llama-cli 标准错误读取线程已停止。")


    def listener_callback(self, msg):
        """ASR 消息回调函数：将接收到的 ASR 文本发送给持久化的 llama-cli 进程"""
        received_text = msg.data
        self.get_logger().info(f'Received ASR: "{received_text}"')

        if not self.llama_process or self.llama_process.poll() is not None:
            self.get_logger().error("llama-cli 进程未运行或已终止。无法处理 ASR 消息。")
            self.start_llama_cli_persistent() # 尝试重新启动进程
            return

        # 在发送新 prompt 之前，清除事件，表示我们期待一个新的回复
        self.response_received_event.clear()
        self.is_first_response_processed = False # 重置标记，准备接收新的第一个回复

        # 构造发送给 llama-cli 的prompt内容，按照Qwen的聊天模板格式
        prompt_content_for_interactive = f"<|im_start|>user\n{received_text}<|im_end|>\n<|im_start|>assistant\n"
        if not prompt_content_for_interactive.endswith('\n'):
            prompt_content_for_interactive += '\n'

        try:
            self.llama_process.stdin.write(prompt_content_for_interactive)
            self.llama_process.stdin.flush()
            self.get_logger().info(f"已将 prompt 发送至 llama-cli 标准输入。")

        except BrokenPipeError:
            self.get_logger().error("llama-cli 标准输入管道已损坏。进程可能已终止。")
            self.llama_process = None
        except Exception as e:
            self.get_logger().error(f"写入 llama-cli 标准输入时发生错误: {e}")

    def process_llm_output_queue(self):
        """定时器回调函数：处理从 LLM 输出队列中收集到的回复并发布"""
        # 只有当一个完整的回复被标记为已就绪时才进行处理
        if self.response_received_event.is_set():
            try:
                # 从队列中获取一个完整的回复
                final_response = self.output_queue.get(timeout=0.1)
                self.response_received_event.clear() # 处理完后清除事件，等待下一个回复

                if not final_response:
                    final_response = "未能生成有效的回复。"

                self.get_logger().info(f"LLM 生成的回复 (控制台输出): \n{final_response}")

                msg_to_publish = String()
                msg_to_publish.data = final_response
                self.publisher_.publish(msg_to_publish)
                self.get_logger().info(f'正在发布 LLM 回复到 /llm_response: "{final_response}"')

            except queue.Empty:
                self.get_logger().warn("response_received_event 已设置，但 output_queue 为空。")
                self.response_received_event.clear()
            except Exception as e:
                self.get_logger().error(f"处理 LLM 输出队列时发生错误: {e}")


    def destroy_node(self):
        """节点销毁时执行的清理操作"""
        self.get_logger().info("正在关闭 LLM Subscriber 节点。")
        self.stop_event.set()
        if hasattr(self, 'stdout_reader_thread') and self.stdout_reader_thread.is_alive():
            self.stdout_reader_thread.join(timeout=1.0)
        if hasattr(self, 'stderr_reader_thread') and self.stderr_reader_thread.is_alive():
            self.stderr_reader_thread.join(timeout=1.0)

        if self.llama_process:
            self.get_logger().info("正在终止 llama-cli 进程...")
            try:
                self.llama_process.terminate()
                self.llama_process.wait(timeout=5)
                if self.llama_process.poll() is None:
                    self.llama_process.kill()
                    self.llama_process.wait(timeout=5)
            except Exception as e:
                self.get_logger().error(f"终止 llama-cli 进程时发生错误: {e}")
            self.get_logger().info("llama-cli 进程已终止。")

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    llm_subscriber = LlmSubscriber()
    try:
        rclpy.spin(llm_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        llm_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()