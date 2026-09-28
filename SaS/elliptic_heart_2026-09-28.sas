/* elliptic_heart_2026-09-28.sas
   Py14aK / SaS
   Construction: Wicklin 2024-02-07, from Conway 2023 and Newton.
   Distinct from SaS/Heart Shaped Box, which uses
   H(x,y) = (x**2 + y**2 - 1)**3 - x**2 * y**3.

   Ellipse used:
       x**2 + y**2 - x*y = 1
   Polar:
       r**2 = 1 / (1 - 0.5*sin(2*theta))
   Heart boundary:
       take the right half of that ellipse, t in [-pi/2, pi/2],
       reflect (x,y) -> (-x,y) with polar angle pi - t,
       sort by theta, draw as one POLYGON.
*/

ods graphics / reset;

%let maxR = 1.4142;
%let minR = 0.8165;

data textOnly;
   retain tx ty 0 msg "ELLIPTIC/HEART";
run;

title "Two tilted ellipses (Conway overlay)";
proc sgplot data=textOnly aspect=1 noautolegend;
   ellipseparm semimajor=&maxR semiminor=&minR /
      slope=1  lineattrs=(color=red thickness=2pt) nofill;
   ellipseparm semimajor=&maxR semiminor=&minR /
      slope=-1 lineattrs=(color=red thickness=2pt) nofill;
   text x=tx y=ty text=msg /
      textattrs=(size=18pt color=red)
      splitchar='/' splitpolicy=splitalways contributeoffsets=none;
   xaxis display=none;
   yaxis display=none;
run;

data Heart;
   retain ID 1;
   pi = constant('pi');
   do t = -pi/2 to pi/2 by pi/200;
      r = sqrt(1 / (1 - 0.5*sin(2*t)));
      y = r*sin(t);
      theta = t;
      x = r*cos(theta);
      output;
      theta = pi - t;
      x = -x;
      output;
   end;
   drop pi t r;
run;

proc sort data=Heart;
   by theta;
run;

data All;
   set textOnly Heart;
run;

title "Elliptic heart (parameterized boundary)";
proc sgplot data=All noborder nowall noautolegend aspect=1;
   polygon x=x y=y ID=ID / fill fillattrs=(color=cxC3540C);
   text x=tx y=ty text=msg /
      textattrs=(size=18pt color=white)
      splitchar='/' splitpolicy=splitalways contributeoffsets=none;
   xaxis display=none;
   yaxis display=none;
run;

title "Elliptic heart with generating ellipses";
proc sgplot data=All aspect=1 noautolegend;
   polygon x=x y=y ID=ID / fill fillattrs=(color=pink);
   ellipseparm semimajor=&maxR semiminor=&minR /
      slope=1  lineattrs=(color=lightred) nofill transparency=0.8;
   ellipseparm semimajor=&maxR semiminor=&minR /
      slope=-1 lineattrs=(color=lightred) nofill transparency=0.8;
   text x=tx y=ty text=msg /
      textattrs=(size=18pt color=red)
      splitchar='/' splitpolicy=splitalways contributeoffsets=none;
   xaxis grid;
   yaxis grid;
run;
