#include "cvwld.h"

CvWLDParams cvWLDParams(int T, int M, int S)
{
	CvWLDParams params;
	params.T = T;
	params.M = M;
	params.S = S;
	return params;
}

CvMat* cvExtractWLD(const CvArr* image, CvWLDParams params)
{
	CvMat imghdr, *img = cvGetMat(image, &imghdr);
	CvMat* hist = cvCreateMat(params.M, params.S * params.T, CV_32SC1);
	cvZero(hist);
	uchar* ptr = img->data.ptr + img->step + 1;
	int i, j;
	for (i = 1; i < img->rows - 1; i++)
	{
		for (j = 1; j < img->cols - 1; j++)
		{
			float sigma = cvFastArctan(ptr[-img->step - 1] + ptr[-img->step] + ptr[-img->step + 1] + ptr[-1] + ptr[1] + ptr[img->step - 1] + ptr[img->step] + ptr[img->step + 1] - 8 * ptr[0], ptr[0]);
			if (sigma > 180.0)
				sigma -= 180;
			float theta = cvFastArctan(ptr[img->step] - ptr[-img->step], ptr[-1] - ptr[1]);
			int t = (int)(theta / 360.0 * params.T);
			int c = (int)(sigma / 180.0 * params.M * params.S);
			hist->data.i[t * params.S + (c % params.S) + (int)(c / params.S) * params.S * params.T]++;
			ptr++;
		}
		ptr += img->step - img->cols + 2;
	}
	CvMat* hist_norm = cvCreateMat(params.M, params.S * params.T, CV_32FC1);
	cvConvertScale(hist, hist_norm, 1.0 / (double)((img->rows - 2) * (img->cols - 2)));
	cvReleaseMat(&hist);
	return hist_norm;
}

void cvExtractWLD(const CvArr* image, CvMemStorage* storage, CvSeq** keypoint, CvSeq** descriptor, CvWLDParams params)
{
}
