#ifndef _TRANSITION_H_
#define _TRANSITION_H_

#include <Zaffre/headers/Zaffre.hpp>

struct Transition {
	int state, action, statePrime;
	double reward;
	bool isInitialState;

    friend std::istream& operator>>(std::istream& str, Transition& data)
    {
        std::string line;
        Transition tmp;
	char chomp;
//        if (std::getline(str,line))
//        {
//            std::stringstream iss(line);
            if ( str >> chomp && str >> tmp.state && str >> chomp && str >> tmp.action && str >> chomp && str >> tmp.reward && str >> chomp ) 
		//std::getline(iss, tmp.state, ',')        && 
                 //std::getline(iss, tmp.action, ',')         &&
                 //std::getline(iss, tmp.reward, ')'))
             {
		 if (chomp == ',')
                   str >> tmp.statePrime;
                 /* OK: All read operations worked */
                 data.swap(tmp);  // C++03 as this answer was written a long time ago.
             }
             else
             {
                 // One operation failed.
                 // So set the state on the main stream
                 // to indicate failure.
                 str.setstate(std::ios::failbit);
             }
//        }
        return str;
    }
    void swap(Transition& other)// throws() // C++03 as this answer was written a long time ago.
    {
	state = other.state;
	action = other.action;
	reward = other.reward;
	statePrime = other.statePrime;
	isInitialState = false;
    }

};

#endif
