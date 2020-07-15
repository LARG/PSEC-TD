#include <Zaffre/headers/Zaffre.hpp>

void init(bool use_tinv)
{
	initParallel();					// Tell Eigen that this application might be parallel.
	if (use_tinv)
		init_tinv();
}
void close(bool clearTempFiles)
{
	if (clearTempFiles)
	{
		const char * Zaffre_Path = getenv("ZAFFRE_PATH");
		if (Zaffre_Path == nullptr) {
			errorExit("Error getting environment variable ZAFFRE_PATH.");
		}
		ofstream out((string)((string)Zaffre_Path + "/temp/linePlot.temp").c_str());
		(void)(system(((string)"(cd " + (string)Zaffre_Path + (string)"/temp; sh clean.sh)").c_str())+1);
	}
}
