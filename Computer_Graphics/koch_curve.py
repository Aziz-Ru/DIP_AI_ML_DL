import turtle
import math
import time
def koch(pen,start,end,depth):
  if depth==0:
    pen.penup()
    pen.goto(start)
    pen.pendown()
    pen.goto(end)
    return
  dx = (end[0]-start[0])/3
  dy = (end[1]-start[1])/3
  p1 = (start[0]+dx,start[1]+dy)
  p3 = (start[0]+ 2*dx,start[1]+ 2*dy)
  px = p1[0]+ (p3[0]-p1[0])/2 + math.sqrt(3)* (p3[1]-p1[1])/2
  py = p1[1]+ (p3[1]-p1[1])/2 - math.sqrt(3)* (p3[0]-p1[0])/2
  p2= (px,py)
  koch(pen, start,p1,depth-1)
  koch(pen,p1,p2,depth-1)
  koch(pen,p2,p3,depth-1)
  koch(pen,p3,end,depth-1)


def draw(pen,vertices,depth):
  pen.pencolor('red')
  
  for i in range(3):
    koch(pen,vertices[i],vertices[(i+1)%3],depth)


s = turtle.Screen()
s.setup(800,800)
pen = turtle.Turtle()
pen.speed(0)

triangle = [(0,150), (-130,-75), (130,-75)]

for i in range(5):  # iterations 0-4
    pen.clear()
    pen.penup()
    pen.goto(-350,280)
    pen.pencolor("black")
    pen.pensize(3)
    pen.write(f"Koch Snowflake - Iteration {i}", font=("Arial",18,"bold"))
    pen.goto(-350,250)
    pen.write("Starting triangle" if i==0 else f"Depth {i}", font=("Arial",12,"normal"))
    draw(pen, triangle, i)
    
    time.sleep(2 if i>0 else 3)

s.mainloop()