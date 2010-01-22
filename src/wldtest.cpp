#include "cvwld.h"
#include "highgui.h"
#include <assert.h>
#include <stdio.h>

int main(int argc, char** argv)
{
	assert(argc == 2);
	IplImage* image = cvLoadImage(argv[1]);
	double t = (double)cvGetTickCount();
	CvMat* hr = cvExtractWLD(image, cvWLDParams(8, 6, 20));
	t = (double)cvGetTickCount() - t;
	printf("extract time = %gms\n", t / ((double)cvGetTickFrequency() * 1000.));
	CvMat* result = cvCreateMat(100, 20 * 6 * 8, CV_8UC3);
	cvNamedWindow("result", CV_WINDOW_AUTOSIZE);
	for (int i = 0; i < 20 * 6 * 8; i++)
		cvLine(result, cvPoint(i, 100 - hr->data.fl[i] * 10000), cvPoint(i, 100), cvScalar(0, 255, 0));
	cvShowImage("result", result);
	cvWaitKey(0);
	cvReleaseMat(&hr);
	cvReleaseMat(&result);
	cvReleaseImage(&image);
	cvDestroyWindow("result");
	return 0;
}
