#include "cvwld.h"
#include "highgui.h"
#include "cvaux.h"

CvWLDParams cvWLDParams(int T, int M, int S, int threshold, CvSize size)
{
	CvWLDParams params;
	params.T = T;
	params.M = M;
	params.S = S;
	params.threshold = threshold;
	params.size = size;
	return params;
}

CvWLDPoint cvWLDPoint(int x, int y, float scale, float dir)
{
	CvWLDPoint kp;
	kp.pt.x = x;
	kp.pt.y = y;
	kp.scale = scale;
	kp.dir = dir;
	return kp;
}

CvMat* cvExtractWLD( const CvArr* image, CvWLDParams params )
{
	CvMat imghdr, *img = cvGetMat( image, &imghdr );
	CvMat* hist = cvCreateMat( params.M, params.S * params.T, CV_32SC1 );
	cvZero( hist );
	uchar* ptr = img->data.ptr + img->step + 1;
	int i, j;
	for ( i = 1; i < img->rows - 1; i++ )
	{
		for ( j = 1; j < img->cols - 1; j++ )
		{
			float sigma = cvFastArctan( ptr[-img->step - 1] + ptr[-img->step] + ptr[-img->step + 1] + ptr[-1] + ptr[1] + ptr[img->step - 1] + ptr[img->step] + ptr[img->step + 1] - 8 * ptr[0], ptr[0] );
			if ( sigma > 180.0 )
				sigma = sigma - 360.0;
			float theta = cvFastArctan( ptr[img->step] - ptr[-img->step], ptr[-1] - ptr[1] );
			int t = (int)( theta / 360.0 * params.T );
			int c = (int)( (sigma + 90.0) / 180.0 * params.M * params.S );
			hist->data.i[t * params.S + (c % params.S) + (int)(c / params.S) * params.S * params.T]++;
			ptr++;
		}
		ptr += img->step - img->cols + 2;
	}
	CvMat* hist_norm = cvCreateMat( params.M, params.S * params.T, CV_32FC1 );
	cvConvertScale( hist, hist_norm, 1.0 / (double)((img->rows - 2) * (img->cols - 2)) );
	cvReleaseMat( &hist );
	return hist_norm;
}

void cvExtractWLD( const CvArr* image, CvMemStorage* storage, CvSeq** _keypoints, CvSeq** _descriptors, CvWLDParams params )
{
	CvMat imghdr, *img = cvGetMat(image, &imghdr);

	int hr = img->rows / (params.size.height * 2);
	int wr = img->cols / (params.size.width * 2);
	int scale_upto = (int)( log( (double)MIN( hr, wr ) ) / log( sqrt(2.) ) );
	/* generae scale-down images */
	CvMat** pyr = (CvMat**)cvStackAlloc( scale_upto * sizeof(pyr[0]) );
	pyr[0] = img;
	double sqrt_2 = sqrt(2.);
	pyr[1] = cvCreateMat( (int)(pyr[0]->rows / sqrt_2), (int)(pyr[0]->cols / sqrt_2), CV_8UC1 );
	cvResize( pyr[0], pyr[1], CV_INTER_AREA );
	int i, j, k, t, x, y;
	for ( i = 2; i < scale_upto; i += 2 )
	{
		pyr[i] = cvCreateMat( pyr[i - 2]->rows >> 1, pyr[i - 2]->cols >> 1, CV_8UC1 );
		cvPyrDown( pyr[i - 2], pyr[i] );
	}
	for ( i = 3; i < scale_upto; i += 2 )
	{
		pyr[i] = cvCreateMat( pyr[i - 2]->rows >> 1, pyr[i - 2]->cols >> 1, CV_8UC1 );
		cvPyrDown( pyr[i - 2], pyr[i] );
	}
	CvSeq* keypoints = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvWLDPoint), storage );
	double scale = 1.0;
	for ( i = 0; i < scale_upto; i++ )
	{
		cv::vector<cv::KeyPoint> tp;
		cv::FAST( pyr[i], tp, params.threshold, true );
		for ( cv::vector<cv::KeyPoint>::iterator it = tp.begin(); it != tp.end(); it++)
		{
			CvWLDPoint kp = cvWLDPoint(it->pt.x, it->pt.y, scale, 0);
			cvSeqPush( keypoints, &kp );
		}
		scale *= sqrt(2.0);
	}
	for ( i = 1; i < scale_upto; i++ )
		cvReleaseMat( &pyr[i] );
	double trfp[6];
	CvMat trf = cvMat(2, 3, CV_64FC1, trfp);
	CvSeq* descriptors = cvCreateSeq( 0, sizeof(CvSeq), sizeof(float) * params.T * params.M * params.S, storage );
	for ( i = 0; i < keypoints->total; i++ )
	{
		CvWLDPoint* kp = *(CvWLDPoint**)cvGetSeqElem( keypoints, i );
		CvMat* lp = cvCreateMat(kp->scale * params.size.height, kp->scale * params.size.width, CV_8UC1);
		float old_dir = kp->dir;
		for (;;)
		{
			cvmSet(&trf, 0, 0, cos(kp->dir) * kp->scale);
			cvmSet(&trf, 0, 1, sin(kp->dir) * kp->scale);
			cvmSet(&trf, 1, 0, -sin(kp->dir) * kp->scale);
			cvmSet(&trf, 1, 1, cos(kp->dir) * kp->scale);
			cvmSet(&trf, 0, 2, kp->pt.x);
			cvmSet(&trf, 1, 2, kp->pt.y);
			cvGetQuadrangleSubPix(img, lp, &trf);
			CvMat* desc = cvExtractWLD( lp, params );
			double radius = 0;
			for ( k = 0; k < params.T * params.S * params.M; k++ )
				radius += desc->data.fl[k];
			radius -= (int)(radius / (3.1415926 * 2));
			kp->dir = radius;
			cvReleaseMat( &lp );
			printf("%f\n", kp->dir);
			if ( fabs(kp->dir - old_dir) < 1e-3 )
				break;
		}
		cvWaitKey(0);
	}
	*_keypoints = keypoints;
	*_descriptors = descriptors;
}
