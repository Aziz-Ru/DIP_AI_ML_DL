#import section
import turtle
import time
import math

#screen section
screen = turtle.Screen()
screen.title("2D transformation")
screen.setup(1000,800)
screen.setworldcoordinates(0,0,1000,800)

#pen section
t = turtle.Turtle()
t.speed(0)
t.penup()
t.hideturtle()

#draw_section
def draw_shape(points, color):
	t.pencolor(color)
	t.pensize(3)
	t.penup()
	for i, (x,y) in enumerate(points):
		t.goto(x,y)
		t.dot(4, color)
		if i==0 :
			t.pendown()
		else :
			t.goto(x,y)
	t.goto(points[0])
	screen.update()
	
#draw axis
t.pencolor("gray"); t.pensize(1); t.penup(); t.goto(-400,0); t.pendown(); t.goto(400,0); t.penup()
t.pencolor("gray"); t.pensize(1); t.penup(); t.goto(0,-400); t.pendown(); t.goto(0,400); t.penup()
t.pencolor("gray"); t.pensize(1); t.penup(); t.goto(-400,-400); t.pendown(); t.goto(400,400); t.penup()

#apply translation
def translation(x,y,tx,ty):
	return x+tx, y+tx


#Rotation
def rotation(x,y, angle_deg, cx,cy):
	angle_rad = math.radians(angle_deg)
	sin_a, cos_a = math.sin(angle_rad), math.cos(angle_rad)
	x-= cx
	y-=cy
	x_new, y_new = x*cos_a - y*sin_a, x*sin_a+y*cos_a
	return x_new+cx, y_new+cx
#scaling
def scaling(x,y,sx,sy,cx,cy):
	x-=cx
	y-=cy
	x_new=x*sx
	y_new=y*sy
	return x_new+cx, y_new+cy

#draw triangle
original = [(-40,-40), (40,-40), (0,60)]
base_x, base_y = 0,0
triangle = [(base_x+x, base_y+y) for x,y in original]
draw_shape(triangle, "black")
time.sleep(1.5)

#Center of the triangle


#draw translated traiangle
translated = [(translation(x,y,200,0)) for x,y in triangle]
draw_shape(translated, "blue")
time.sleep(1.5)
#Center
cx = sum(p[0] for p in translated)/3

cy = sum(p[1] for p in translated)/3
#Rotation
rotated = [rotation(x,y,45, cx, cy) for x,y in translated]
draw_shape(rotated, "red")
time.sleep(1.5)

#Scaling
scaled = [scaling(x,y,1.5,1.5,cx,cy) for x,y in rotated]
draw_shape(scaled, "purple")
screen.exitonclick()