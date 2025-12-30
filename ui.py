import tkinter as tk
from tkinter import ttk
import re
from config import Config
class AppUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Control Panel")
        self.root.geometry("600x450")
        
        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # IP Address input
        ttk.Label(main_frame, text="IP Address:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.ip_entry = ttk.Entry(main_frame, width=20)
        self.ip_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        self.ip_entry.insert(0, "127.0.0.1")  # Default localhost
        
        # Rounds dropdown
        ttk.Label(main_frame, text="Rounds:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.rounds_var = tk.StringVar(value="1")  # Default value 1
        self.rounds_dropdown = ttk.Combobox(main_frame, textvariable=self.rounds_var, 
                                           values=[str(i) for i in range(1, 11)], 
                                           state="readonly", width=18)
        self.rounds_dropdown.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # No. of clients dropdown
        ttk.Label(main_frame, text="No. of Clients:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.clients_var = tk.StringVar(value="1")  # Default value 1
        self.clients_dropdown = ttk.Combobox(main_frame, textvariable=self.clients_var, 
                                            values=[str(i) for i in range(1, 11)], 
                                            state="readonly", width=18)
        self.clients_dropdown.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Button frame for Start and Start Fresh
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Start button
        self.start_button = ttk.Button(button_frame, text="Start", command=self.on_start)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Start Fresh button
        self.start_fresh_button = ttk.Button(button_frame, text="Start Fresh", 
                                            command=self.on_start_fresh)
        self.start_fresh_button.pack(side=tk.LEFT, padx=5)
        
        # Display window (Text widget with scrollbar)
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(display_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text widget with scroll
        self.display_text = tk.Text(display_frame, wrap=tk.WORD, 
                                   yscrollcommand=scrollbar.set,
                                   height=15, width=50)
        self.display_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.display_text.yview)
    
    # Helper functions for display window
    def display(self, text):
        """Display text in the display window"""
        self.display_text.insert(tk.END, text + "\n")
        self.display_text.see(tk.END)  # Auto-scroll to bottom
    
    def clear(self):
        """Clear all text from display window"""
        self.display_text.delete(1.0, tk.END)
    
    def append(self, text):
        """Append text without newline"""
        self.display_text.insert(tk.END, text)
        self.display_text.see(tk.END)
    
    def get_display_content(self):
        """Get all text from display window"""
        return self.display_text.get(1.0, tk.END)
    
    # Getter functions
    def get_ip(self):
        """Get the current IP address from entry"""
        return self.ip_entry.get()
    
    def get_rounds(self):
        """Get the selected rounds from dropdown"""
        return int(self.rounds_var.get())
    
    def get_clients(self):
        """Get the selected number of clients from dropdown"""
        return int(self.clients_var.get())
    
    def validate_ip(self):
        """Validate IPv4 address format"""
        ip = self.get_ip()
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(pattern, ip):
            parts = ip.split('.')
            return all(0 <= int(part) <= 255 for part in parts)
        return False
    
    def on_start(self):
        from server import start

        """Called when Start button is clicked"""
        if not self.validate_ip():
            self.display("Error: Invalid IP address format!")
            return
        
        ip = self.get_ip()
        rounds = self.get_rounds()
        clients = self.get_clients()
        self.display(f"Started with IP: {ip}, Rounds: {rounds}, Clients: {clients}")
        Config.HOST=ip
        Config.ROUNDS=rounds
        Config.NUM_CLIENTS=clients
        start()
        # Add your custom logic here
    
    def on_start_fresh(self):
        """Called when Start Fresh button is clicked"""
        if not self.validate_ip():
            self.display("Error: Invalid IP address format!")
            return
        
        ip = self.get_ip()
        rounds = self.get_rounds()
        clients = self.get_clients()
        self.display(f"Starting Fresh with IP: {ip}, Rounds: {rounds}, Clients: {clients}")
        
        # Add your custom fresh start logic here

def main():
    root = tk.Tk()
    app = AppUI(root)
    


    
    root.mainloop()

if __name__ == "__main__":
    main()