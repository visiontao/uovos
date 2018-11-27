#ifndef _GaussianPyramid_h
#define _GaussianPyramid_h

#include "Image.h"

class GaussianPyramid
{
public:
  GaussianPyramid();
  ~GaussianPyramid();

  void ConstructPyramid(const DImage& image,
                        double ratio = 0.8, int minWidth = 30);

  int nlevels() const { return nLevels; }
  DImage& Image(int index) { return ImPyramid[index]; }

private:
  DImage* ImPyramid;
  int nLevels;
};

#endif
