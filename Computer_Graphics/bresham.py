import turtle

s = turtle.Screen()
s.title("Bresenham's Line Drawing Algorithm")
s.bgcolor('white')
s.setup(width=1000, height=800)
s.setworldcoordinates(0,0,1000,800)

t = turtle.Turtle()
t.speed(2)
t.penup()

def draw_line(points):
  for i, (x,y) in enumerate(points):
    if i == 0:
      t.goto(x, y)
      t.pendown()
    else:
      t.goto(x, y)
  t.penup()

def bresham(x1,y1,x2,y2):
  points =[]

  dx = abs(x2-x1)
  dy = abs(y2-y1)

  # direction of increment
  sx = 1 if dx>0 else -1
  sy = 1 if dy>0 else -1

  if dx >dy:
    p = 2*dy -dx
    x = x1; y=y1
    for i in range(dx+1):
      points.append((x,y))
      x+=sx
      if p>=0:
        y+=sy
        p-= 2*dx
      
      p+= 2*dy
  else:
    p = 2*dx-dy
    x = x1; y=y1
    for i in range(dy+1):
      points.append((x,y))
      y+=sy
      if p>=0:
        x+=sx
        p-= 2*dy
      
      p+= 2*dx
  return points


points1 = bresham(100, 300, 400, 600)
points2 = bresham(100, 600, 800, 500)
