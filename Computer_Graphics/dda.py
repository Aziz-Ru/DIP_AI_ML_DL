import turtle

screen = turtle.Screen()
screen.title("DDA Line Drawing Algorithm")
screen.bgcolor('white')
screen.setup(width=1000, height=800)
screen.setworldcoordinates(0,0,1000,800)
# screen.setworldcoordinates(0,0,1000,800)

t = turtle.Turtle()
t.speed(2)
t.penup()
t.pencolor('blue')
t.pensize(2)

def draw_line(points):
  for i, (x,y) in enumerate(points):
    if i == 0:
      t.goto(x, y)
      t.pendown()
    else:
      t.goto(x, y)
  t.penup()
  


def dda(x1,y1,x2,y2):
  dx = x2-x1
  dy = y2-y1
  steps = max(abs(dx), abs(dy))
  x_inc = dx/ steps
  y_inc = dy/ steps
  x = x1
  y = y1
  points =[]
  for i in range(steps+1):
    points.append((round(x), round(y)))
    x += x_inc
    y += y_inc
  return points


points1 = dda(100, 300, 400, 600)
points2 = dda(100, 600, 800, 500)
draw_line(points1)
draw_line(points2)

screen.mainloop()
