import tkinter as tk
from tkinter import ttk
from transformers import pipeline
import torch

# Charger le modèle et le tokenizer
model_name = "distilgpt2-finetuned-bio"  # Assurez-vous que le chemin est correct
generator = pipeline('text-generation', model=model_name, device=0 if torch.cuda.is_available() else -1)

class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chat avec DistilGPT-2")
        self.geometry("400x300")

        self.entry_var = tk.StringVar()
        self.entry = ttk.Entry(self, textvariable=self.entry_var, width=40)
        self.entry.pack(pady=5)

        self.submit_button = ttk.Button(self, text="Envoyer", command=self.on_submit)
        self.submit_button.pack(pady=5)

        self.text_widget = tk.Text(self, wrap=tk.WORD, height=10, width=50)
        self.text_widget.insert(tk.END, "Entrez votre texte ici...\n")
        self.text_widget.pack(expand=True, fill=tk.BOTH)

    def on_submit(self):
        text = self.entry_var.get().strip()
        if not text:
            return

        # Générer la réponse
        result = generator(text, max_length=50, num_return_sequences=1)[0]['generated_text']
        self.text_widget.insert(tk.END, f"\nRéponse: {result}\n")

if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
