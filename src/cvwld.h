#ifndef _GUARD_cvwld_h_
#define _GUARD_cvwld_h_

#include <cv.h>

typedef struct CvWLDParams {
	int T, M, S;
} CvWLDParams;

CvWLDParams cvWLDParams(int T, int M, int S);
CvMat* cvExtractWLD(const CvArr* image, CvWLDParams params);
void cvExtractWLD(const CvArr* image, CvMemStorage* storage, CvSeq** keypoint, CvSeq** descriptor, CvWLDParams params);

#endif
