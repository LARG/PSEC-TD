#ifndef _ZAFFRE_TIMER_H_
#define _ZAFFRE_TIMER_H_

/*
Code for timing how long something takes.
*/

#include <time.h>

class Timer {
public:
	Timer();
	void start();
	float getMS();		// Get the number of milliseconds since start was called
	float getS();		// Get the number of seconds since start was called
	float getM();		// Get the number of minutes since start was called
	float getH();		// Get the number of hours since start was called
private:
	clock_t startTime;
};

#endif
