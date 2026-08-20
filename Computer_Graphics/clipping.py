import turtle
screen = turtle.Screen()

screen.title("Cohen-Sutherland Line Clipping Algorithm")
screen.bgcolor('white')
screen.setup(width=1000, height=800)
screen.setworldcoordinates(0,0,1000,800)


t = turtle.Turtle()
t.penup()

X_MIN, X_MAX = 100, 300
Y_MIN, Y_MAX = 200, 400

INSIDE = 0
LEFT = 1
RIGHT = 2
BOTTOM = 4
TOP = 8

def draw_line(x1, y1, x2, y2,color):
    t.pencolor(color)
    t.goto(x1,y1)
    t.pendown()
    t.goto(x2,y2)
    t.penup()

def compute_code(x, y):
    code = INSIDE
    if x<X_MIN:
        code |= LEFT
    elif x>X_MAX:
        code |=RIGHT
    if y<Y_MIN:
        code |= BOTTOM
    elif y>Y_MAX:
        code |= TOP
    return code

def sutherland_clip(x1,y1,x2,y2):
    code1 = compute_code(x1,y1)
    code2 = compute_code(x2,y2)
    

    while True:
        if code1==0 and code2==0:
            draw_line(x1,y1,x2,y2,'blue')
            break
        elif code1&code2:
            break
        else:
            code_out = code1 if code1 else code2
            if code_out & TOP:
                x =x1+ (x2-x1)*(Y_MAX-y1)/(y2-y1)
                y = Y_MAX
            elif code_out & BOTTOM:
                x = x1 + (x2-x1)*(Y_MIN-y1)/(y2-y1)
                y = Y_MIN
            elif code_out & LEFT:
                x= X_MIN
                y = y1 +(y2-y1)*(X_MIN-x1)/(x2-x1)
            elif code_out & RIGHT:
                x = X_MAX
                y = y1 + (y2-y1)*(X_MAX-x1)/(x2-x1)
            if code_out == code1:
                x1,y1 = x,y
                code1 = compute_code(x1,y1)
            else:
                x2,y2 = x,y
                code2 = compute_code(x2,y2)

draw_line(X_MIN, Y_MIN, X_MAX, Y_MIN,'green')
draw_line(X_MIN, Y_MAX, X_MIN, Y_MIN,'green')
draw_line(X_MAX, Y_MIN, X_MAX, Y_MAX,'green')
draw_line(X_MAX, Y_MAX, X_MIN, Y_MAX,'green')

x1,y1,x2,y2 = 50,150, 350, 450
draw_line(x1,y1,x2,y2,'red')
sutherland_clip(x1,y1,x2,y2)
screen.mainloop()