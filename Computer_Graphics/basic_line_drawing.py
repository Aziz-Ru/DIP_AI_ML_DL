import turtle
import math
screen = turtle.Screen() 
screen.title("Basic Line Drawing")
screen.bgcolor('white')
screen.setup(width=1000, height=800)
screen.setworldcoordinates(0,0,1000,800)

# drawing cursor
cursor = turtle.Turtle()
# Turtle-এর animation speed। Lowest speed is 1, fastest speed is 10, and 0 means no animation.
cursor.speed(0)
# Pen উঠিয়ে দিচ্ছে। 
cursor.penup()
# Turtle-এর ছোট arrow/cursor-টা hide করবে।
cursor.hideturtle()

def draw_shape(points):
  cursor.penup()
  for i,(x,y) in enumerate(points):
    if i == 0:
      cursor.goto(x, y)
      cursor.pendown()
    else:
      cursor.goto(x, y)
  cursor.goto(points[0])
  cursor.penup()

def draw_line(x1, y1, x2, y2):
  # this starting point
  cursor.pensize(3)
  cursor.pencolor('red')
  cursor.goto(x1, y1)
  # now put the pen down to start drawing
  cursor.pendown()
  # move the cursor to the ending point
  cursor.goto(x2, y2)
  # lift the pen up to stop drawing
  cursor.penup()
  # screen.update()  # Update the screen to show the drawn line

# cursor.pencolor('grey');cursor.pensize(1);cursor.penup();cursor.goto(-400,0);cursor.pendown();cursor.goto(400,0);cursor.penup()
# cursor.pencolor('grey');cursor.pensize(1);cursor.penup();cursor.goto(0,-400);cursor.pendown();cursor.goto(0,400);cursor.penup()
# cursor.pencolor('grey');cursor.pensize(1);cursor.penup();cursor.goto(-400,-400);cursor.pendown();cursor.goto(400,400);cursor.penup()



# draw_line(100, 600, 400, 800)
# draw_line(10, 500, 300, 100)

# draw_line(0, 400, 800, 400)

def translation(points, tx,ty):
  new_points = []
  for x,y in points:
    new_x = x + tx
    new_y = y + ty
    new_points.append((new_x, new_y))
  return new_points

def scaling(points, sx, sy,cx,cy):
  new_points = []
  for x,y in points:
    x-=cx
    y-=cy
    new_x = x * sx
    new_y = y * sy
    new_points.append((new_x+cx, new_y+cy))
  return new_points
  

def rotation(points, angle,cx,cy):
  radians = math.radians(angle)
  s = math.sin(radians)
  c = math.cos(radians)
  new_points = []

  for x,y in points:
    x-=cx
    y-=cy
    new_x = x * c - y * s
    new_y = x * s + y * c
    new_points.append((new_x+cx, new_y+cy))
  return new_points

# shape1 = [(200, 200), (800, 200), (800, 600), (200, 600)]
# draw_shape(shape1)

# shape2 = translation(shape1, -100, -100)
# draw_shape(shape2)

# shape3 = scaling(shape1, 0.1, 0.3)
# draw_shape(shape3)

# shape4 = rotation(shape1, math.pi / 4)
# draw_shape(shape4)

cursor.pencolor('red');cursor.pensize(2); cursor.goto(0, 0);cursor.pendown();cursor.goto(800, 0);cursor.penup()
cursor.pencolor('red');cursor.pensize(2); cursor.goto(0, 0);cursor.pendown();cursor.goto(0, 800);cursor.penup()
cursor.pencolor('red');cursor.pensize(2); cursor.goto(0, 0);cursor.pendown();cursor.goto(800, 800);cursor.penup()


triangle = [(50, 100), (150, 100), (75, 200)]
cursor.pencolor('blue')
draw_shape(triangle)

cx = sum(p[0] for p in triangle)/3
cy = sum(p[1] for p in triangle)/3

translate = translation(triangle, 300, 300)
draw_shape(translate)

trinagle3 = scaling(triangle, 1.5, 1.5,cx,cy)
draw_shape(trinagle3)

trinagle4 = rotation(triangle, 45,cx,cy)
draw_shape(trinagle4)
# Keep the window open until the user clicks on it
screen.mainloop()




