import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black

def create_pdf_from_files(files, output_pdf):
    # Create a PDF canvas
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter

    # Set up initial position for text
    x, y = 50, height - 50
    line_height = 12

    # Function to write a title with extra space
    def write_title(title):
        nonlocal y
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(black)
        c.drawString(x, y, f"--- {title} ---")
        y -= 20  # Space after title

    # Loop through each file
    for file_path in files:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            
            # Write the title of the file (e.g., model.py, server.py)
            write_title(f"File: {filename}")
            
            # Read the content of the file
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Write file content
            c.setFont("Courier", 10)
            for line in lines:
                c.drawString(x, y, line.strip())
                y -= line_height
                if y < 50:  # Check if we need to create a new page
                    c.showPage()
                    y = height - 50
                    # Add space before the next section
                    write_title(f"File: {filename}")
            
            y -= 20  # Add some space between files

    # Save the PDF
    c.save()

# Example usage:
folder_path = "app"  # The folder containing the files
files_to_include = [
    os.path.join(folder_path, "model.py"),
    os.path.join(folder_path, "server.py"),
    os.path.join(folder_path, "index.tsx")
]
output_pdf = "output.pdf"
create_pdf_from_files(files_to_include, output_pdf)
