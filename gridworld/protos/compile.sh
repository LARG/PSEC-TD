rm ../headers/results.pb.h
rm ../src/results.pb.cpp
rm ../scripts/results_pb2.py
/usr/bin/protoc results.proto --python_out=.
/usr/bin/protoc results.proto --cpp_out=.
mv results.pb.h ../headers/
mv results.pb.cc ../src/results.pb.cpp
#mv results_pb2.py ../scripts
