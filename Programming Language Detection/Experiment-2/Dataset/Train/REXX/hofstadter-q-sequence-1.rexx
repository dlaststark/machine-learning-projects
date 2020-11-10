/*REXX program generates the    Hofstadter  Q     sequence for any specified   N.       */
parse arg a b c d .                              /*obtain optional arguments from the CL*/
if a=='' | a==","  then a=       10              /*Not specified?  Then use the default.*/
if b=='' | b==","  then b=    -1000              /* "      "         "   "   "      "   */
if c=='' | c==","  then c=  -100000              /* "      "         "   "   "      "   */
if d=='' | d==","  then d= -1000000              /* "      "         "   "   "      "   */
q.= 1;                 ac=   abs(c)              /* [↑]  negative #'s don't show values.*/
call HofstadterQ  a
call HofstadterQ  b;   say;    say  abs(b)th(b)      'value is:'      result;          say
call HofstadterQ  c
downs= 0;              do j=2  for ac-1;        jm= j - 1
                          downs= downs + (q.j<q.jm)
                       end   /*j*/

say downs  'terms are less then the previous term,'    ac || th(ac)    'term is:'     q.ac
call HofstadterQ  d;                     ad= abs(d);            say
say 'The'      ad || th(ad)        'term is'           q.ad
exit                                             /*stick a fork in it,  we're all done. */
/*──────────────────────────────────────────────────────────────────────────────────────*/
HofstadterQ: procedure expose q.; parse arg x 1 ox     /*get number to generate through.*/
                                                       /* [↑]   OX    is the same as X. */
x= abs(x)                                              /*use the absolute value for  X. */
w= length(x)                                           /*use for right justified output.*/
             do j=1  for x                             /* [↓]  use short─circuit IF test*/
             if j>2   then if q.j==1  then  do;    jm1= j - 1;             jm2= j - 2
                                                    _1= j - q.jm1;          _2= j - q.jm2
                                                   q.j= q._1  +  q._2
                                            end
             if ox>0  then say right(j,w) right(q.j,w) /*display the number if  OX > 0. */
             end    /*j*/
return q.x                                             /*return the │X│th term to caller*/
/*──────────────────────────────────────────────────────────────────────────────────────*/
th: procedure; x=abs(arg(1)); return word('th st nd rd',1+x//10*(x//100%10\==1)*(x//10<4))
