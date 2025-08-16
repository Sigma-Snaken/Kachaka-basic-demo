import io
import kachaka_api
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk
from kachaka_api.generated import kachaka_api_pb2 as pb2
from kachaka_api.util.vision import OBJECT_LABEL, get_bbox_drawn_image
import cv2
import asyncio
import math
import tkinter as tk
from tkinter import ttk
import traceback

# 將IP位址定義為常數，方便未來修改
KACHAKA_TARGET_IP = "192.168.50.102:26400"

def world_to_pixel(world_x: float, world_y: float, map_info: pb2.Map):
    """將世界座標 (公尺) 轉換為地圖像素座標。"""
    pixel_x = (world_x - map_info.origin.x) / map_info.resolution
    # 影像座標的 Y 軸通常與世界座標相反 (原點在左上角)
    pixel_y = map_info.height - ((world_y - map_info.origin.y) / map_info.resolution)
    return pixel_x, pixel_y

class KachakaDashboardApp(tk.Tk):
    def __init__(self, client: kachaka_api.aio.KachakaApiClient, command_queue: asyncio.Queue, robot_is_idle_event: asyncio.Event):
        super().__init__()
        self.client = client
        self.command_queue = command_queue
        self.robot_is_idle_event = robot_is_idle_event
        self.running = True

        self.title("Kachaka 機器人儀表板")
        self.geometry("1600x900")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False

        # --- UI 佈局 ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 設定 ttk 元件的樣式，加大字體
        style = ttk.Style(self)
        # 使用系統上可能存在且支援中文的字體
        large_font = ('Microsoft JhengHei', 14, 'bold')
        style.configure('TButton', font=large_font, padding=10)
        style.configure('TCombobox', font=large_font)
        # 設定下拉選單中的字體大小
        self.option_add('*TCombobox*Listbox.font', ('Microsoft JhengHei', 12))

        # 左側地圖
        map_frame = ttk.Frame(main_frame)
        map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # 右側攝影機與控制
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 設定 right_frame 的 grid layout，讓攝影機畫面均分空間
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)  # 前鏡頭畫面
        right_frame.rowconfigure(2, weight=1)  # 後鏡頭畫面

        control_frame = ttk.Frame(right_frame)
        control_frame.grid(row=0, column=0, sticky="ew", pady=5)

        front_cam_frame = ttk.LabelFrame(right_frame, text="前鏡頭 (含物件偵測)")
        front_cam_frame.grid(row=1, column=0, sticky="nsew", pady=5)

        back_cam_frame = ttk.LabelFrame(right_frame, text="後鏡頭")
        back_cam_frame.grid(row=2, column=0, sticky="nsew", pady=5)

        info_frame = ttk.LabelFrame(right_frame, text="偵測到的物件")
        info_frame.grid(row=3, column=0, sticky="ew", pady=5)
        self.info_label = ttk.Label(info_frame, text="", justify=tk.LEFT)
        self.info_label.pack(anchor="w")

        # --- Matplotlib 圖表設定 ---
        self.fig = plt.Figure(figsize=(8, 8), dpi=100)
        self.ax_map = self.fig.add_subplot(111)
        self.ax_map.set_title("地圖 & 即時位置")
        self.ax_map.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=map_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Tkinter 控制元件 ---
        self.location_var = tk.StringVar()
        self.location_dropdown = ttk.Combobox(control_frame, textvariable=self.location_var, state="readonly")
        self.location_dropdown.pack(side=tk.LEFT, padx=5)
        
        self.go_button = ttk.Button(control_frame, text="GO", command=self.on_go_button_clicked)
        self.go_button.pack(side=tk.LEFT, padx=5)

        self.return_home_button = ttk.Button(control_frame, text="回充電站", command=self.on_return_home_button_clicked)
        self.return_home_button.pack(side=tk.LEFT, padx=5)

        # --- 預先建立的 Artist ---
        self.front_img_display = ttk.Label(front_cam_frame)
        self.front_img_display.pack()
        self.back_img_display = ttk.Label(back_cam_frame)
        self.back_img_display.pack()
        self.robot_arrow_artist = None

        # --- 影片錄製設定 ---
        self.front_video_writer = None
        self.back_video_writer = None
        self.combined_video_writer = None

    def on_closing(self):
        self.running = False
        print("\n視窗已關閉，正在停止程式...")

    def on_go_button_clicked(self):
        selected_location = self.location_var.get()
        if not selected_location:
            print("請先選擇一個地點。")
            return

        if not self.robot_is_idle_event.is_set():
            print("指令無法傳送：機器人正在移動中。")
            return
        
        print(f"按鈕已點擊！傳送移動到 '{selected_location}' 的指令。")
        self.command_queue.put_nowait(selected_location)

    def on_return_home_button_clicked(self):
        if not self.robot_is_idle_event.is_set():
            print("指令無法傳送：機器人正在移動中。")
            return
        
        print("按鈕已點擊！傳送返回充電站的指令。")
        self.command_queue.put_nowait("return_home")

    def resize_pil_image(self, pil_image, widget):
        """將 PIL 影像縮放以符合 Tkinter 元件的大小，同時保持長寬比。"""
        container = widget.master
        # 減去一些邊距，避免影像緊貼邊框
        max_w = container.winfo_width() - 20
        max_h = container.winfo_height() - 40 # 為 LabelFrame 的標題保留額外空間

        if max_w > 1 and max_h > 1:
            # 使用 thumbnail 來縮放影像，它會保持長寬比
            pil_image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        return pil_image

    def pil_to_tk(self, pil_image):
        """將 PIL 影像轉換為 Tkinter PhotoImage"""
        return ImageTk.PhotoImage(image=pil_image, master=self)

    async def initialize_data(self):
        """獲取地圖和地點等初始靜態資料"""
        print("正在獲取初始資料 (地圖、地點)...")
        try:
            self.map_info = await self.client.get_png_map()
            image_data_stream = io.BytesIO(self.map_info.data)
            pil_image = Image.open(image_data_stream)
            self.map_img_array = np.array(pil_image)
            
            locations = await self.client.get_locations()
            self.location_names = [loc.name for loc in locations]
            print(f"成功獲取地圖 '{self.map_info.name}' 及 {len(locations)} 個地點。")

            self.ax_map.imshow(self.map_img_array)
            for loc in locations:
                px, py = world_to_pixel(loc.pose.x, loc.pose.y, self.map_info)
                self.ax_map.scatter(px, py, c='blue', s=100, alpha=0.7)
                self.ax_map.text(px + 10, py, loc.name, color='blue', fontsize=12)
            
            if self.location_names:
                self.location_dropdown['values'] = self.location_names
                self.location_dropdown.current(0)
            else:
                print("警告: 未找到任何地點。")

            # 初始化錄影
            first_front_image = await anext(self.client.front_camera_ros_compressed_image.stream())
            if first_front_image.data:
                pil_img = Image.open(io.BytesIO(first_front_image.data))
                width, height = pil_img.size
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.front_video_writer = cv2.VideoWriter('front_camera_record.mp4', fourcc, 10.0, (width, height))
                print(f"開始錄製前鏡頭畫面到 'front_camera_record.mp4' (尺寸: {width}x{height})")

        except Exception as e:
            print(f"獲取初始資料時發生錯誤: {e}")
            self.on_closing()

    async def update_loop(self):
        """主要的非同步更新迴圈"""
        front_cam_stream = self.client.front_camera_ros_compressed_image.stream()
        object_detection_stream = self.client.object_detection.stream()

        while self.running:
            try:
                # --- 更新機器人位置 ---
                robot_pose = await self.client.get_robot_pose()
                px, py = world_to_pixel(robot_pose.x, robot_pose.y, self.map_info)
                if self.robot_arrow_artist:
                    self.robot_arrow_artist.remove()
                arrow_len = 25
                dx = arrow_len * math.cos(robot_pose.theta)
                dy = -arrow_len * math.sin(robot_pose.theta)
                self.robot_arrow_artist = self.ax_map.arrow(px, py, dx, dy, head_width=10, head_length=10, fc='red', ec='red')

                # --- 同時獲取攝影機影像和物件偵測結果 ---
                front_cam_image_ros, back_cam_image_ros, detection_result = await asyncio.gather(
                    anext(front_cam_stream),
                    self.client.get_back_camera_ros_compressed_image(),
                    anext(object_detection_stream)
                )
                (header, objects) = detection_result

                # --- 準備原始 PIL 影像 ---
                front_img_pil = Image.open(io.BytesIO(front_cam_image_ros.data))
                back_img_pil = Image.open(io.BytesIO(back_cam_image_ros.data)) if back_cam_image_ros.data else None

                # --- 根據 frame_id 決定在哪個影像上繪製 bounding box ---
                if header.frame_id == "camera_front":
                    front_img_pil_with_bbox = get_bbox_drawn_image(front_cam_image_ros, objects)
                    front_img_to_display = front_img_pil_with_bbox
                    back_img_to_display = back_img_pil
                elif back_img_pil and header.frame_id == "camera_back":
                    back_img_pil_with_bbox = get_bbox_drawn_image(back_cam_image_ros, objects)
                    front_img_to_display = front_img_pil
                    back_img_to_display = back_img_pil_with_bbox
                else:
                    front_img_to_display = front_img_pil
                    back_img_to_display = back_img_pil

                # --- 更新後鏡頭畫面 ---
                if back_img_to_display:
                    back_img_resized = self.resize_pil_image(back_img_to_display.copy(), self.back_img_display)
                    back_img_tk = self.pil_to_tk(back_img_resized)
                    self.back_img_display.configure(image=back_img_tk)
                    self.back_img_display.image = back_img_tk

                    back_img_array = np.array(back_img_to_display)
                    if self.back_video_writer is None:
                        height, width, _ = back_img_array.shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.back_video_writer = cv2.VideoWriter('back_camera_record.mp4', fourcc, 10.0, (width, height))
                        print(f"開始錄製後鏡頭畫面到 'back_camera_record.mp4' (尺寸: {width}x{height})")
                    
                    if self.back_video_writer:
                        frame_bgr = cv2.cvtColor(back_img_array, cv2.COLOR_RGB2BGR)
                        self.back_video_writer.write(frame_bgr)

                # --- 更新前鏡頭畫面 ---
                front_img_resized = self.resize_pil_image(front_img_to_display.copy(), self.front_img_display)
                front_img_tk = self.pil_to_tk(front_img_resized)
                self.front_img_display.configure(image=front_img_tk)
                self.front_img_display.image = front_img_tk

                if self.front_video_writer:
                    frame_bgr = cv2.cvtColor(np.array(front_img_to_display), cv2.COLOR_RGB2BGR)
                    self.front_video_writer.write(frame_bgr)

                # --- 合併錄影 ---
                if front_img_to_display and back_img_to_display:
                    front_array = np.array(front_img_to_display)
                    back_array = np.array(back_img_to_display)

                    if front_array.shape[1] == back_array.shape[1]:
                        if self.combined_video_writer is None:
                            height = front_array.shape[0] + back_array.shape[0]
                            width = front_array.shape[1]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            self.combined_video_writer = cv2.VideoWriter('combined.mp4', fourcc, 10.0, (width, height))
                            print(f"開始錄製合併畫面到 'combined.mp4' (尺寸: {width}x{height})")

                        if self.combined_video_writer:
                            combined_frame = np.vstack((front_array, back_array))
                            frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
                            self.combined_video_writer.write(frame_bgr)

                # --- 更新物件資訊 ---
                info_text = "\n".join([f"- {OBJECT_LABEL[obj.label]} ({obj.score:.2f})" for obj in objects])
                self.info_label.config(text=info_text)

                # --- 重繪 Matplotlib & 更新 Tkinter ---
                self.canvas.draw_idle()
                self.update()
                self.update_idletasks()
                
                await asyncio.sleep(0.1)

            except (asyncio.CancelledError, tk.TclError):
                break # 當視窗關閉時會觸發 TclError
            except Exception as e:
                print(f"更新迴圈中發生錯誤: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

        # --- 清理 ---
        if self.front_video_writer:
            self.front_video_writer.release()
            print("前鏡頭錄影已儲存並關閉。")
        if self.back_video_writer:
            self.back_video_writer.release()
            print("後鏡頭錄影已儲存並關閉。")
        if self.combined_video_writer:
            self.combined_video_writer.release()
            print("合併錄影已儲存並關閉。")

async def run_robot_commands_async(client: kachaka_api.aio.KachakaApiClient, command_queue: asyncio.Queue, robot_is_idle_event: asyncio.Event):
    """
    一個等待佇列中的移動指令並執行的背景任務。
    """
    try:
        while True:
            command = await command_queue.get()
            robot_is_idle_event.clear()
            
            try:
                if command == "return_home":
                    print("收到返回充電站的指令...")
                    result = await client.return_home()
                    if result.success:
                        print("已成功返回充電站。")
                    else:
                        print(f"返回充電站失敗: {result.error_code}, {result.message}")
                else: # This is a move_to_location command
                    target_location = command
                    print(f"收到指令，正在移動到 '{target_location}'...")
                    await client.update_resolver()
                    result = await client.move_to_location(target_location)
                    if result.success:
                        print(f"已成功抵達 '{target_location}'。")
                    else:
                        print(f"移動到 '{target_location}' 失敗: {result.error_code}, {result.message}")
            except Exception as e:
                print(f"執行指令 '{command}' 時發生例外: {e}")
                traceback.print_exc()

            robot_is_idle_event.set()
            command_queue.task_done()
    except asyncio.CancelledError:
        print("移動指令任務已取消。")

async def main():
    """
    主函式，用於連接到 Kachaka 機器人，顯示其地圖，
    並可選擇性地執行命令。
    """
    client = None
    app = None
    robot_task = None
    try:
        client = kachaka_api.aio.KachakaApiClient(target=KACHAKA_TARGET_IP)
        command_queue = asyncio.Queue()
        robot_is_idle_event = asyncio.Event()
        robot_is_idle_event.set()

        print("正在啟動儀表板與機器人移動任務...")
        app = KachakaDashboardApp(client, command_queue, robot_is_idle_event)

        # 將機器人指令和 UI 更新作為並行任務執行
        robot_task = asyncio.create_task(
            run_robot_commands_async(client, command_queue, robot_is_idle_event)
        )
        
        await app.initialize_data()
        
        # update_loop 是主要的 UI 迴圈，會一直執行直到視窗關閉
        await app.update_loop()

    except Exception as e:
        print(f"程式執行時發生錯誤: {e}")
        traceback.print_exc()
    finally:
        if robot_task:
            robot_task.cancel()
        if client:
            print("正在關閉與機器的連線...")
            # await client.close()
            print("連線已關閉。")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass