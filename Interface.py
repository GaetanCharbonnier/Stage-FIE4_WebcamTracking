from tkinter import * 
import os
import tkinter

def open_DistanceFaceWbc():
    os.startfile("FaceWbc_4.py")

def open_DistanceFaceWbc2():
    os.startfile("FaceWbcGame2.py")

def open_GraphDist_day():
    os.startfile("GraphDist.py")

def open_GraphDist_month():
    os.startfile("GraphDistMonth.py")

def open_Login(): 
    os.startfile("Login.py")
    main_interface.destroy()
    
def openWindowTest():
  open_DistanceFaceWbc()

def openGameTest():
  open_DistanceFaceWbc2()

def Rcap_day():
  open_GraphDist_day()

def Rcap_month():
  open_GraphDist_month()

def about_help():

    screen1 = Tk()
    screen1.title("About WebTrack")

    width_value = screen1.winfo_screenwidth()
    height_value = screen1.winfo_screenheight()
    screen1.geometry("%dx%d+0+0" % (width_value, height_value))
    
    screen1.iconbitmap("Icon/sitdown.ico")
    screen1.config(background= '#41B77F')

    # Création de la frame 
    frame = Frame(screen1, bg='#41B77F')

    # Ajouter un texte 
    label_title = Label(frame, text="Security", font=("Courrier", 20), bg='#41B77F', fg='white')
    label_title.pack(expand=YES)
  
    # Ajouter un texte 
    label_subtitle = Label(frame, text="Thank for your interest in safe!", font=("Courrier", 10), bg='#41B77F', fg='white')
    label_subtitle.pack(expand=YES)
    
    frame.pack(expand=YES)

def main_interface():
  # Création première fenetre
  window = Tk()
 
  # Personnalisation fenetre
  window.title("WebTrack")

  width_value = window.winfo_screenwidth()
  height_value = window.winfo_screenheight()
  window.geometry("%dx%d+0+0" % (width_value, height_value))

  window.iconbitmap("Icon/sitdown.ico")
  window.config(background= '#41B77F')
  
  # Création de la frame 
  # frame = Frame(window, bg='#41B77F')

  # frame 1
  Frame1 = Frame(window, borderwidth=2, relief=GROOVE, bg='#41B77F')
  Frame1.pack(padx=30, pady=30)

  # Ajouter un texte 
  label_title = Label(Frame1, text="Welcome to WebTrack", font=("Courrier", 40), bg='#41B77F', fg='white')
  label_title.pack(expand=YES)

  #  Ajouter un texte 
  label_subtitle = Label(Frame1, text="Preston Project", font=("Courrier", 15), bg='#41B77F', fg='white')
  label_subtitle.pack(expand=YES)

  # Ajouter un bouton
  Web_button = Button(Frame1, text="Open WebTrack", font=("Courrier", 20), bg='white', fg='#41B77F', command=openWindowTest)
  Web_button.pack(padx=20, pady=20, fill=X)

  # Ajouter un bouton
  Web_button = Button(Frame1, text="Open Start_WebTrack", font=("Courrier", 20), bg='white', fg='#41B77F', command=openGameTest)
  Web_button.pack(padx=20, pady=20, fill=X)

  # Ajouter un bouton
  Web_button = Button(Frame1, text="Recap of the day", font=("Courrier", 20), bg='white', fg='#41B77F', command=Rcap_day)
  Web_button.pack(padx=20, pady=10, side=tkinter.LEFT)

    # Ajouter un bouton
  Web_button = Button(Frame1, text="Recap of the month", font=("Courrier", 20), bg='white', fg='#41B77F', command=Rcap_month)
  Web_button.pack(padx=20, pady=10, side=tkinter.RIGHT)

  Frame1.pack(expand=YES)

  # Menubar
  menubar = Menu(window)

  menu1 = Menu(menubar, tearoff=0)
  menu1.add_command(label="Disconnect", command=lambda:[window.quit(), open_Login()])
  menu1.add_separator()
  menu1.add_command(label="Close", command=window.quit)
  menubar.add_cascade(label="File", menu=menu1)

  menu2 = Menu(menubar, tearoff=0)
  menu2.add_command(label="About WebTrack", command=about_help)
  menubar.add_cascade(label="Help", menu=menu2)

  window.config(menu=menubar)

  #afficher la fenetre
  window.mainloop()

main_interface()
