#include <Zaffre/headers/Zaffre.hpp>

Timer::Timer()
{
	start();	// Might as well start now anyway - so the first start call is optional
}

void Timer::start()
{
	startTime = clock();
}

float Timer::getMS()
{
	return 1000.0f * (float)(clock() - startTime) / (CLOCKS_PER_SEC);
}

float Timer::getS()
{
	return (float)(clock() - startTime) / (CLOCKS_PER_SEC);
}

float Timer::getM()
{
	return (float)(clock() - startTime) / (60.0f * CLOCKS_PER_SEC);
}

float Timer::getH()
{
	return (float)(clock() - startTime) / (3600.0f * CLOCKS_PER_SEC);
}
