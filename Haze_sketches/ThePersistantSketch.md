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


-- we will continue this later --- 

Category theory. 			