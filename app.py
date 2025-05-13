import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QSlider, QPushButton, QLineEdit
 
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Load tokenizer and model
model_dir = "Bambii-03/arabic-summarize" # hugging face repo 
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
 
# Custom summarization function
def summarize_paragraph(paragraph, ratio=0.47):
    tokens = tokenizer.encode_plus(
        paragraph,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
 
    target_length = max(1, int(ratio * len(tokenizer.tokenize(paragraph))))
 
    summary_tokens = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=target_length
    )
    summary = tokenizer.decode(summary_tokens[0], skip_special_tokens=True)
    return summary
 
# PyQt5 Application
class SummarizerApp(QWidget):
    def _init_(self):
        super()._init_()
 
        self.setWindowTitle("Arabic Text Summarizer")
        self.setGeometry(100, 100, 600, 500)  # Adjust window size
 
        # Layouts
        layout = QVBoxLayout()
        input_layout = QHBoxLayout()
 
        # Input Textbox
        self.text_input = QTextEdit(self)
        self.text_input.setPlaceholderText("Enter Arabic Text")
        self.text_input.setFixedHeight(100)
        layout.addWidget(self.text_input)
 
        # Slider for summary ratio
        self.slider = QSlider(self)
        self.slider.setOrientation(1)  # Horizontal orientation
        self.slider.setMinimum(10)
        self.slider.setMaximum(100)
        self.slider.setValue(47)  # Default value
        self.slider.setTickInterval(1)
 
        self.ratio_label = QLabel(f"Summary Ratio: {self.slider.value()}%", self)
        self.slider.valueChanged.connect(self.update_ratio_label)  # Connect slider value change to label update
 
        input_layout.addWidget(self.ratio_label)
        input_layout.addWidget(self.slider)
 
        # Summary Output (changed to QTextEdit for multi-line display)
        self.summary_output = QTextEdit(self)  # Use QTextEdit instead of QLineEdit
        self.summary_output.setReadOnly(True)
        self.summary_output.setPlaceholderText("Summary will appear here...")
        self.summary_output.setFixedHeight(150)  # Increase height of summary output box
        layout.addLayout(input_layout)
        layout.addWidget(self.summary_output)
 
        # Summarize Button
        self.summarize_button = QPushButton('Summarize', self)
        self.summarize_button.clicked.connect(self.on_summarize_click)
        layout.addWidget(self.summarize_button)
 
        self.setLayout(layout)
 
    def update_ratio_label(self):
        # Update the ratio label when the slider value changes
        self.ratio_label.setText(f"Summary Ratio: {self.slider.value()}%")
 
    def on_summarize_click(self):
        paragraph = self.text_input.toPlainText()
        ratio = self.slider.value() / 100  # Convert slider value to ratio between 0.1 and 1.0
        summary = summarize_paragraph(paragraph, ratio)
        self.summary_output.setText(summary)
 
# Main function to run the app
if _name_ == "_main_":
    app = QApplication(sys.argv)
    window = SummarizerApp()
    window.show()
    sys.exit(app.exec_())