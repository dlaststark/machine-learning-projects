/*REXX program  tests  if a number  (or a range of numbers)  is/are  perfect.           */
parse arg low high .                             /*obtain the specified number(s).      */
if high=='' & low==""  then high=34000000        /*if no arguments,  then use a range.  */
if  low==''            then  low=1               /*if no   LOW,  then assume unity.     */
if high==''            then high=low             /*if no  HIGH,  then assume  LOW.      */
w=length(high)                                   /*use  W  for formatting the output.   */
numeric digits max(9,w+2)                        /*ensure enough digits to handle number*/

             do i=low  to high                   /*process the single number or a range.*/
             if isPerfect(i)  then say  right(i,w)  'is a perfect number.'
             end   /*i*/
exit                                             /*stick a fork in it,  we're all done. */
/*──────────────────────────────────────────────────────────────────────────────────────*/
isPerfect: procedure;  parse arg x 1 y           /*obtain the number to be tested.      */
           if x==6  then return 1                /*handle the special case of  six.     */
                                                 /*[↓]  perfect number's digitalRoot = 1*/
                 do  until  y<10                 /*find the digital root of  Y.         */
                 parse var y r 2;   do k=2  for length(y)-1; r=r+substr(y,k,1); end  /*k*/
                 y=r                             /*find digital root of the digit root. */
                 end   /*until*/                 /*wash, rinse, repeat ···              */

           if r\==1  then return 0               /*Digital root ¬ 1?   Then  ¬ perfect. */
           s=1                                   /*the first factor of  X.           ___*/
                       do j=2  while  j*j<=x     /*starting at 2, find the factors ≤√ X */
                       if x//j\==0  then iterate /*J  isn't a factor of X,  so skip it. */
                       s = s + j + x%j           /*··· add it  and  the other factor.   */
                       end   /*j*/               /*(above)  is marginally faster.       */
           return s==x                           /*if the sum matches  X, it's perfect! */
