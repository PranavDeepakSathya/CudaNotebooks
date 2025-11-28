Okay, Welcome. 

So, let us build a quick barrier model. A barrier B is a tuple (Ts, C) where Ts is the set of threads 
Participating in the barrier B, and C is the expected arrival count. A thread t in the barrier B can be in the following states: 0 -> pre_arive, 1 -> arrive, 2-> post_arrive, 3 -> wait_sync, 4 -> post_wait_sync. 
ko
So those are the 5 stages of sync right, now what does docs of cuda say? But before that some math. 
let |N depict the set of natural numbers,for say clock cycles, 
then, the State of a barrier s(B) is a functon s: IDXs{Ts} x |N --> {0,1,2,3,4}^|Ts| x [C] x |N 
that is, the state takes the tuple of thread indices of Ts, the current clock cycle, and returns the 
tuple of the state of the threads in the barrier, the number of threads that have arrived, 

now let me just make a table for about four threads, we assume that ones a thread arrives, it is in post arrival 
just in the next clock cycle itself. 
t0 t1 t2 t3  c_ariv  b_phase
0  0  0  0		0		0
0  0  0  0 		0		0
1  0  1  0		2		0
2  0  2  1		3		0
3  0  2  2      3       0
3  0  3  3      3       0 
3  0  3  3      3       0
3  1  3  3      4       0  --all threads have arrived
4  2  4  4		0		1  -- new barrier phase 


for a thread t, t moves from stage 0 to state 1 when t calls token(t) <- bar.arrive(t). 
and then immediately moves to stage 2, (post arrive) (a thread is unblocked when it calls arrive, however the count 
of the barrier decrememnts) Next, to move to the wait stage, a thread calls block(t) <- bar.wait(token(t))


One must call bar.arrive only when the barriers counter is non zero. 

If the call of bar.arrive finally makes the counter zero, then the bar.wait must be called beffore using the barrier 
again for arrival. 

let the phase of the barrier be $p$, bar.wait(token(t)) where token holds the current phase, only makes sense 
if token(t) = p or p - 1. nothing else. 

Producer-consumer. 

Imagine I have an assembly line: 

p0===========|b0 |======c0
p1===========|b1 |======c1

where the left half is the producer (with say two lines ) and the right half is the consumer (again two lines)

the consumer c can itself consume from at most one line, and the producer p itself can atmost produce to one line 

initially both b0, b1 are empty. 

The producer and consumer are very strict people, and act in a very safe manner. 

at initalization, the p says. Hey I don't know if these buffers are empty to c. and then 
1. p waits for c to say a buffer is empty
2. c says hey, b0 is empty
3. c waits for p to fill b0
4. p decides to fill b0
5. p0 fills b0 
6. p says to c, b0 is filled and waits for the next instruction
7. c says to p, b1 is not filled
8. p1 starts filling b1, meanwhile c0 starts working on b0 
9. p1 finishes filling b1 and signals to c that b1 is full and then waits 
10. c0 finishes working on b0 and then immediately signals that b0 is empty 
11. p0 starts filling b0 meanwhile c1 works on b1. 

Monoidal Preorder (P, =<, e, o) where P, =< is a pre order relation (reflexive and transitive on P)
and (P,e,o) is a monoid, for which when the diagonal order (p,p') =< (q,q') iff p =< p' and q =< q' 
on PxP, induces o : PxP --> P to be a monotone map. ie p =< p' and q =< q' implies p o q <= p' o q'
