OPENCV = /opt/opencv
CXX = g++
LINKFLAG = -pthread -lcv -lhighgui -lcvaux
INCFLAGS = -I"$(OPENCV)/include/opencv"
CXXFLAGS = -O2 --fast-math -msse2

all: wldtest

clean:
	rm *.o ../lib/*.o

wldtest: wldtest.o cvwld.o
	$(CXX) -o $@ $^ $(LINKFLAG)

.c.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cpp.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cc.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)
