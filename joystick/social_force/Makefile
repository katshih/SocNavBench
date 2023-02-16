CXX = g++
RM = rm -f
INCLUDES = -I./src
LIBS= src/libpedsim.a
CXXFLAGS = ${INCLUDES} -std=c++17 -g -O0
PROGRAMS = social_force example01

all: $(PROGRAMS) src/libpedsim.a

src/libpedsim.a:
	$(MAKE) -C src clean all

social_force: social_force.cpp src/libpedsim.a
	$(RM) social_force
	$(CXX) social_force.cpp $(CXXFLAGS) -o social_force $(LIBS)

example01: example01.cpp
	$(RM) example01
	$(CXX) example01.cpp $(CXXFLAGS) -o example01 $(LIBS)

clean:
	$(RM) $(PROGRAMS)
	$(RM) *.o
	$(RM) src/libpedsim.a

.PHONY: all clean
