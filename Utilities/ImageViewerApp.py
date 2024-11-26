import os
import threading
import queue
from tkinter import Tk, Label, Frame, Canvas, Scrollbar, Button, filedialog
from PIL import Image, ImageTk


class ImageViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")

        # Main layout
        self.image_frame = Frame(root, width=800, height=800)  # Fixed size for image display
        self.image_frame.pack(side='left', fill='both', expand=True)

        self.scroll_frame = Frame(root, width=400)  # Locked width for the side pane
        self.scroll_frame.pack(side='right', fill='y')

        # Canvas for image display
        self.image_label = Label(self.image_frame, text="Select an image folder", bg="gray")
        self.image_label.pack(fill='both', expand=True)

        # Scrollable frame for thumbnails
        self.canvas = Canvas(self.scroll_frame, width=400)  # Match the locked width
        self.scrollbar = Scrollbar(self.scroll_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.scrollable_frame = Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)

        # Folder selection button
        self.folder_button = Button(self.scroll_frame, text="Select Folder", command=self.load_images)
        self.folder_button.pack(side="bottom", fill="x")

        # Thumbnail row/column tracking
        self.thumbnails_per_row = 3  # Fixed number of thumbnails per row
        self.thumbnail_width = 100
        self.thumbnail_spacing = 10  # Space between thumbnails
        self.current_row = 0
        self.current_column = 0

        # Queue for thread-safe thumbnail processing
        self.thumbnail_queue = queue.Queue()

        # Bind scrolling to the entire window
        self.root.bind("<MouseWheel>", self.scroll_canvas)  # For Windows and MacOS
        self.root.bind("<Button-4>", self.scroll_canvas)   # For Linux (Up)
        self.root.bind("<Button-5>", self.scroll_canvas)   # For Linux (Down)

    def scroll_canvas(self, event):
        if event.num == 4 or event.delta > 0:  # Scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:  # Scroll down
            self.canvas.yview_scroll(1, "units")

    def load_images(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        # Clear existing thumbnails
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Reset row/column tracking
        self.current_row = 0
        self.current_column = 0

        # Disable folder selection button while loading
        self.folder_button.config(state="disabled")

        # Collect image paths
        self.image_paths = [
            os.path.join(folder_path, filename)
            for filename in os.listdir(folder_path)
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
        ]

        # Add image paths to the queue
        for image_path in self.image_paths:
            self.thumbnail_queue.put(image_path)

        # Start loading thumbnails in a separate thread
        threading.Thread(target=self.process_thumbnails).start()

    def process_thumbnails(self):
        while not self.thumbnail_queue.empty():
            image_path = self.thumbnail_queue.get()
            try:
                self.add_thumbnail(image_path)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        # Re-enable folder selection button after all thumbnails are loaded
        self.root.after(0, lambda: self.folder_button.config(state="normal"))

    def add_thumbnail(self, image_path):
        try:
            thumbnail_size = (self.thumbnail_width, self.thumbnail_width)
            img = Image.open(image_path)
            img.thumbnail(thumbnail_size)
            img_tk = ImageTk.PhotoImage(img)  # Catch allocation errors here

            # Add thumbnail to the GUI dynamically on the main thread
            self.root.after(0, self._display_thumbnail, img_tk, image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    def _display_thumbnail(self, img_tk, image_path):
        # Create thumbnail label
        thumbnail_label = Label(self.scrollable_frame, image=img_tk)
        thumbnail_label.image = img_tk  # Keep a reference to avoid garbage collection
        thumbnail_label.grid(row=self.current_row, column=self.current_column, padx=self.thumbnail_spacing,
                             pady=self.thumbnail_spacing)
        thumbnail_label.bind("<Button-1>", lambda e, p=image_path: self.display_image(p))

        # Update row/column tracking
        self.current_column += 1
        if self.current_column >= self.thumbnails_per_row:
            self.current_column = 0
            self.current_row += 1

    def display_image(self, image_path):
        try:
            img = Image.open(image_path)

            # Scale image to fit the height of the viewing panel while preserving aspect ratio
            frame_height = self.image_frame.winfo_height()
            width, height = img.size
            new_width = int(width * (frame_height / height))  # Calculate new width based on frame height
            img = img.resize((new_width, frame_height), Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(img)

            self.image_label.configure(image=img_tk, text="")  # Remove text when displaying an image
            self.image_label.image = img_tk  # Keep a reference to avoid garbage collection
        except Exception as e:
            print(f"Error displaying image {image_path}: {e}")


if __name__ == "__main__":
    root = Tk()
    root.state('zoomed')  # Maximizes the window on Windows
    app = ImageViewerApp(root)
    root.mainloop()
