#include "cvwld.h"
#include "highgui.h"
#include <assert.h>
#include <stdio.h>

int main(int argc, char** argv)
{
	assert(argc == 2);
	IplImage* image = cvLoadImage(argv[1]);
	IplImage* gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	cvCvtColor(image, gray, CV_BGR2GRAY);
	CvMemStorage* storage = cvCreateMemStorage(0);
	double t = (double)cvGetTickCount();
	CvSeq* keypoints;
	CvSeq* descriptors;
	// CvMat* hr;
	// cvExtractWLD(gray, storage, &keypoints, &descriptors, cvWLDParams(8, 6, 4, 100, cvSize(64, 64)));
	CvMat* hr = cvExtractWLD(gray, cvWLDParams(8, 6, 20, 100, cvSize(64, 64)));
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
	cvReleaseImage(&gray);
	cvReleaseImage(&image);
	cvDestroyWindow("result");
	return 0;
}
