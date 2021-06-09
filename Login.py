from tkinter import * 
import os

global screen, screen1, screen2, screen4, screen5

def open_Interface():
    os.startfile("Interface.py")

def delete3():
  screen4.destroy()

def delete4():
  screen5.destroy()

def password_not_recognised():
  global screen4
  screen4 = Toplevel(screen2)
  screen4.title("Success")
  screen4.iconbitmap("Icon/sitdown.ico")
  screen4.config(background= '#41B77F')
  screen4.geometry("150x100")

  # Création de la frame 
  frame = Frame(screen4, bg='#41B77F')

  # Ajouter un texte 
  label_title = Label(frame, text="Password Error", font=("Courrier", 15), bg='#41B77F', fg='white')
  label_title.pack(expand=YES)

  # Ajouter un bouton
  OK_button = Button(frame, text="OK", font=("Courrier", 10), bg='white', fg='#41B77F', command=delete3)
  OK_button.pack(pady=20, fill=X)

  frame.pack(expand=YES)

def user_not_found():
  global screen5
  screen5 = Toplevel(screen2)
  screen5.title("Success")
  screen5.geometry("150x100")
  screen5.iconbitmap("Icon/sitdown.ico")
  screen5.config(background= '#41B77F')

  # Création de la frame 
  frame = Frame(screen5, bg='#41B77F')

  # Ajouter un texte 
  label_title = Label(frame, text="User Not Found", font=("Courrier", 10), bg='#41B77F', fg='white')
  label_title.pack(expand=YES)

  # Ajouter un bouton
  OK_button = Button(frame, text="OK", font=("Courrier", 15), bg='white', fg='#41B77F', command=delete4)
  OK_button.pack(pady=20, fill=X)

  frame.pack(expand=YES)

def ReturnHome():
  global screen1, screen
  screen1.destroy()
  main_screen()

def register_user():
    print("working")

    username_info = username.get()
    passward_info = password.get()

    file=open(username_info, "w")
    file.write(username_info+"\n")
    file.write(passward_info)
    file.close()

    username_entry.delete(0, END)
    password_entry.delete(0, END)

    label_RegisSucess = Label(text= "Registration Sucess", font=("Courrier", 15), bg='#41B77F', fg='#8a2be2')
    label_RegisSucess.pack(expand=YES)

def login_verify():
    print("working...")
    username1 = username_verify.get()
    password1 = password_verify.get()
    username_entry1.delete(0, END)
    password_entry1.delete(0, END)

    list_of_files = os.listdir()
    if username1 in list_of_files:
        file1 = open(username1, "r")
        verify = file1.read().splitlines()
        if password1 in verify:
            print("Login Sucess")
            LogTest()
        else:
            password_not_recognised()

    else:
            user_not_found()

def LogTest():
  global screen2
  screen2.destroy()
  open_Interface()

def registerTest():
  global screen, screen1
  screen.destroy()
  register()

def loginTest():
  global screen, screen2
  screen.destroy()
  login()

def register():

    global screen1
    screen1 = Tk()
    screen1.title("register")

    width_value = screen1.winfo_screenwidth()
    height_value = screen1.winfo_screenheight()
    screen1.geometry("%dx%d+0+0" % (width_value, height_value))

    screen1.iconbitmap("Icon/sitdown.ico")
    screen1.config(background= '#41B77F')

    global username
    global password
    global username_entry
    global password_entry
    username = StringVar ()
    password = StringVar ()

    # Création de la frame 
    frame = Frame(screen1, borderwidth=2, relief=GROOVE, bg='#41B77F')

    # Ajouter un texte 
    label_title = Label(frame, text="WebTrack Register", font=("Courrier", 40), bg='#41B77F', fg='white')
    label_title.pack(padx=80, pady=20, expand=YES)
  
    # Ajouter un texte 
    label_subtitle = Label(frame, text="Please enter detail below", font=("Courrier", 20), bg='#41B77F', fg='white')
    label_subtitle.pack(expand=YES)

    label_usertext = Label(frame, text="username * ", font=("Courrier", 15), bg='#41B77F', fg='#8a2be2')
    label_usertext.pack(expand=YES)
    username_entry = Entry(frame, textvariable= username)
    username_entry.pack(expand=YES)

    label_passtext = Label(frame, text="password * ", font=("Courrier", 15), bg='#41B77F', fg='#8a2be2')
    label_passtext.pack(expand=YES)
    password_entry = Entry(frame, textvariable= password)
    password_entry.pack(expand=YES)

    # Ajouter un bouton
    Register_button = Button(frame, text="Register", font=("Courrier", 20), bg='white', fg='#41B77F', command=register_user)
    Register_button.pack(padx=20, pady=25, fill=X)

    # Ajouter un bouton
    Return_button = Button(frame, text="Home Page", font=("Courrier", 20), bg='white', fg='#41B77F' , command=ReturnHome)
    Return_button.pack(padx=20, pady=25, fill=X)
    
    frame.pack(expand=YES)


hidden = True

def login():
  global screen2
  screen2 = Tk()
  screen2.title("login")

  width_value = screen2.winfo_screenwidth()
  height_value = screen2.winfo_screenheight()

  screen2.geometry("%dx%d+0+0" % (width_value, height_value))

  # Mode plein ecran
  #screen2.attributes('-fullscreen', True)
  #screen2.bind("<Escape>",lambda event: screen2.attributes("-fullscreen", False))
  # Empêche le redimensionnement de la fenêtre
  #screen2.resizable(width=False,height=False)
  screen2.iconbitmap("Icon/sitdown.ico")
  screen2.config(background= '#41B77F')

  # Création de la frame 
  frame = Frame(screen2, borderwidth=2, relief=GROOVE, bg='#41B77F')

  # Ajouter un texte 
  label_title = Label(frame, text="WebTrack Login", font=("Courrier", 40), bg='#41B77F', fg='white')
  label_title.pack(side=TOP)

  # Ajouter un texte 
  label_subtitle = Label(frame, text="Please enter detail below to Login", font=("Courrier", 15), bg='#41B77F', fg='white')
  label_subtitle.pack(side=TOP, pady=15)

  global username_verify
  global password_verify

  username_verify = StringVar ()
  password_verify = StringVar ()

  global username_entry1
  global password_entry1

  label_usertext = Label(frame, text="username * ", font=("Courrier", 15), bg='#41B77F', fg='#8a2be2')
  label_usertext.pack()
  username_entry1 = Entry(frame, textvariable= username_verify, bg='lavender')
  username_entry1.pack(pady=10)

  label_passtext = Label(frame, text="password * ", font=("Courrier", 15), bg='#41B77F', fg='#8a2be2' )
  label_passtext.pack()
  password_entry1 = Entry(frame, textvariable= password_verify, bg='lavender', show='●')
  password_entry1.pack(pady=10)
  password_entry1.focus_set()

  def update_entry():
    global hidden
    if hidden:
      password_entry1['show'] = ''
      btn['image'] = hide
    else:
      password_entry1['show'] = '●'
      btn['image'] = view
    hidden = not hidden

  hide = PhotoImage(file='Icon/hide.png')
  view = PhotoImage(file='Icon/view.png')

  btn = Button(frame, image=hide, width=60, height=40, font='Times 15 bold', command=update_entry)
  btn.pack(side=RIGHT,padx=10)

  # Ajouter un bouton
  Login_button = Button(frame, text="Login", font=("Courrier", 20), bg='white', fg='#41B77F', command=login_verify)
  Login_button.pack(fill=X, padx=20, pady=20, side=BOTTOM)
  
  frame.pack(expand=YES)
  print("login session started")
  
def main_screen():
    global screen
    screen = Tk()
    screen.title("WebTrack")

    width_value = screen.winfo_screenwidth()
    height_value = screen.winfo_screenheight()
    screen.geometry("%dx%d+0+0" % (width_value, height_value))

    screen.iconbitmap("Icon/sitdown.ico")
    screen.config(background= '#41B77F')

    # Création de la frame 
    frame = Frame(screen, borderwidth=2, relief=GROOVE, bg='#41B77F')

    # Ajouter un texte 
    label_title = Label(frame, text="Welcome to WebTrack", font=("Courrier", 40), bg='#41B77F', fg='white')
    label_title.pack(padx=60, pady=20, expand=YES)
    
    # Ajouter un texte 
    label_subtitle = Label(frame, text="Preston Project", font=("Courrier", 15), bg='#41B77F', fg='white')
    label_subtitle.pack(expand=YES)

    # Ajouter un bouton
    Login_button = Button(frame, text="Login", font=("Courrier", 20), bg='white', fg='#41B77F', command=loginTest)
    Login_button.pack(padx=20, pady=20, fill=X)

    # Ajouter un bouton
    Register_button = Button(frame, text="Register", font=("Courrier", 20), bg='white', fg='#41B77F', command=registerTest)
    Register_button.pack(padx=20, pady=20, fill=X)

    frame.pack(expand=YES)

    screen.mainloop()

main_screen()