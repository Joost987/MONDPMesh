#include <iostream>
#include <vector>
#include <numeric>
#include <Eigen/Dense>
#include <cmath>
#include<complex.h>
#include <array>
#include <fftw3.h>
#include <math.h>
#include <fstream>

using namespace std;


const int halfpixels=64;
float celllen= 1*std::pow(10,10);
float cellleninv=1.0/celllen;
float pi= 3.14159265359;
float oversqrt2pi=1.0/std::sqrt(2*pi);
float oversqrt2pi3=std::pow(oversqrt2pi,3);
float cellvolume=std::pow(celllen,3);
float cellvolumeinv=1/cellvolume;
int scalarsize[3]= {halfpixels*2,halfpixels*2,halfpixels*2};

const float kstep = pi / (halfpixels * celllen);

const float G = 1.0;
const float c = 4 * pi * G;
const float a0 = 1;

const int T = 4;
const int N = 2;

class Particlelist;
class FixedSizeMat2;
class FixedSizeMatVec;
class TwoVecMats;


Eigen::Matrix<float, Eigen::Dynamic, 3> AssignAccsGauss(FixedSizeMatVec accmat, Particlelist particlelist, Eigen::Matrix<float, Eigen::Dynamic, 3> accparts, int N, float sigma=1);
FixedSizeMat2 CalcPot(FixedSizeMat2 Density, fftw_plan pfwd, fftw_plan pbwd); 
FixedSizeMatVec CalcAcc(FixedSizeMat2 Pot, FixedSizeMatVec accmat); 
FixedSizeMat2 inpol(FixedSizeMatVec x, int func); 
FixedSizeMat2 inpolinv(FixedSizeMatVec x, int func); 
TwoVecMats MainLoop(TwoVecMats HNDacc, int func, fftw_plan pfwdvecH, fftw_plan pbwdvecH, int lastit);
FixedSizeMat2 AssignMassGauss(FixedSizeMat2 density, Particlelist particlelist, int N, float sigma=1);

void write_csv(std::string filename, float* mat, int size) {
    std::ofstream myFile(filename);
    for (int i = 0; i < size; ++i) {
        myFile << *(mat + i)<<";";

    }
    myFile.close();
}

class FixedSizeMat2 {
    public:
        fftw_complex* mat;
        FixedSizeMat2() {
            mat = (fftw_complex*)fftw_malloc(2 * halfpixels * 2 * halfpixels * 2 * halfpixels * sizeof(fftw_complex));
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                        mat[k+2*halfpixels*(j+2*halfpixels*i)][0] = 0;
                        mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)][1] = 0;
                    }
                }
            }
        }
        FixedSizeMat2 copy(FixedSizeMat2 A) {
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                        //double b = a * *mat(i, j, k, 'r');
                        A.mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)][0] = mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)][0];

                        // fixMat.mat(i, j, k, 'i') = a * mat(i, j, k, 'i');
                    }
                }
            }
            return A;
        }
        double* operator()(int i, int j, int k,char t) {
            double *val=NULL;
            
            if (t == 'r') {
                val = &mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)][0];
            }
            if (t == 'i') {
                val = &mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)][1];
            }
            
            return val;
        }

        void operator*(float a) {
            //Note: This multiplication is in place
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                        //double b = a * *mat(i, j, k, 'r');
                        mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)][0] *= a;
                        mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)][1] *= a;
                       // fixMat.mat(i, j, k, 'i') = a * mat(i, j, k, 'i');
                    }
                }
            }
          
        }
        void operator*(FixedSizeMat2 A) {
            //Note: This multiplication is in place
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                        mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)][0] *= A.mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)][0];


                    }
                }
            }

        }
        float sum() {
            float res = 0;
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                        res += mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)][0];


                    }
                }
            }
            return res;
        }
        void sqrtinv() {
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                            mat[ (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0] = 1.0 / std::sqrt(mat[ (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0]);
                    }
                }
            }
        }
        void fillzero() {
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                        mat[(k + 2 * halfpixels * (j + 2 * halfpixels * i))][0] = 0;
                        mat[(k + 2 * halfpixels * (j + 2 * halfpixels * i))][1] = 0;
                    }
                }
            }
        }
};

class FixedSizeMatVec {
public:
    fftw_complex* mat;
    FixedSizeMatVec() {
        mat = (fftw_complex*)fftw_malloc(3*2 * halfpixels * 2 * halfpixels * 2 * halfpixels * sizeof(fftw_complex));
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        mat[l+3*(k + 2 * halfpixels * (j + 2 * halfpixels * i))][0] = 0;
                       // mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][1] = 0;
                    }
                }
            }
        }
    }
    
    double* operator()(int i, int j, int k, int l) {
        double* val = NULL;

     
        val = &mat[l+3*( k + 2 * halfpixels * (j + 2 * halfpixels * i))][0];




        return val;
    }

    void operator*(float a) {
        //Note: This multiplication is in place
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        //double b = a * *mat(i, j, k, 'r');
                        mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0] *= a;
                       // mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][1] *= a;
                        // fixMat.mat(i, j, k, 'i') = a * mat(i, j, k, 'i');
                    }
                }
            }
        }

    }

    void operator*(FixedSizeMat2 A) {
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0] *= A.mat[(k + 2 * halfpixels * (j + 2 * halfpixels * i))][0];

                    }
                }
            }
        }
    }

    void operator+(FixedSizeMatVec A) {
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        //double b = a * *mat(i, j, k, 'r');
                        mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0] += A.mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0];
                        // mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][1] *= a;
                         // fixMat.mat(i, j, k, 'i') = a * mat(i, j, k, 'i');
                    }
                }
            }
        }
    }
    void sqrtinv() {
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        //double b = a * *mat(i, j, k, 'r');
                        mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0] = 1.0/std::sqrt(mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0]);
                        // mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][1] *= a;
                         // fixMat.mat(i, j, k, 'i') = a * mat(i, j, k, 'i');
                    }
                }
            }
        }
    }
    FixedSizeMat2 norm() {
        FixedSizeMat2 normmat;
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    double x, y, z;
                    x = mat[0 + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0];
                    y = mat[1 + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0];
                    z = mat[2 + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0];
                    *normmat(i, j, k, 'r') = std::sqrt(x * x + y * y + z * z);

                    
                }
            }
        }
        return normmat;
    }
    void CurlFreeProj(fftw_plan pfwdvec, fftw_plan pbwdvec) {
        fftw_execute(pfwdvec);
        for (int i = -halfpixels; i < halfpixels; ++i) {
            for (int j = -halfpixels; j <  halfpixels; ++j) {
                for (int k = -halfpixels; k < halfpixels; ++k) {

                    double x, y, z, intermediatestep;
                    int currentIndex = 3 * ((k + 2 * halfpixels) % (2 * halfpixels) + 2 * halfpixels * ((j + 2 * halfpixels) % (2 * halfpixels) + 2 * halfpixels * ((i + 2 * halfpixels) % (2 * halfpixels))));
                    //real part;
                    
                    x = mat[0 + currentIndex][0];
                    y = mat[1 + currentIndex][0];
                    z = mat[2 + currentIndex][0];
                    if(i==0 and j==0 and k==0){
                        mat[0 + currentIndex][0] = 0;
                        mat[1 + currentIndex][0] = 0;
                        mat[2 + currentIndex][0] = 0;

                        mat[0 + currentIndex][1] = 0;
                        mat[1 + currentIndex][1] = 0;
                        mat[2 + currentIndex][1] = 0;
                    }
                    else {
                        intermediatestep = (i * x + j * y + k * z) / (i * i + j * j + k * k);
                        mat[0 + currentIndex][0] = i * intermediatestep;
                        mat[1 + currentIndex][0] = j * intermediatestep;
                        mat[2 + currentIndex][0] = k * intermediatestep;

                        //imaginary part
                        x = mat[0 + currentIndex][1];
                        y = mat[1 + currentIndex][1];
                        z = mat[2 + currentIndex][1];
                        intermediatestep = (i * x + j * y + k * z) / (i * i + j * j + k * k);
                        mat[0 + currentIndex][1] = i * intermediatestep;
                        mat[1 + currentIndex][1] = j * intermediatestep;
                        mat[2 + currentIndex][1] = k * intermediatestep;
                    }
                }
            }
        }
        fftw_execute(pbwdvec);
        
    }

    void DivFreeProj(fftw_plan pfwdvec, fftw_plan pbwdvec) {

        fftw_execute(pfwdvec);


        for (int i = -halfpixels; i < halfpixels; ++i) {
           
            for (int j = -halfpixels; j < halfpixels; ++j) {
                for (int k = -halfpixels; k < halfpixels; ++k) {

                    double x, y, z, intermediatestep;
                    int currentIndex = 3 * ((k + 2 * halfpixels) % (2 * halfpixels) + 2 * halfpixels * ((j + 2 * halfpixels) % (2 * halfpixels) + 2 * halfpixels * ((i + 2 * halfpixels) % (2 * halfpixels))));
                    //real part;
                    
                    x = mat[0 + currentIndex][0];
                    y = mat[1 + currentIndex][0];
                    z = mat[2 + currentIndex][0];
                    if (i == 0 and j == 0 and k == 0) {

                       // mat[0 + currentIndex][0] -= 0;
                       // mat[1 + currentIndex][0] -= 0;
                       //mat[2 + currentIndex][0] -= 0;

                       // mat[0 + currentIndex][1] -= 0;
                       // mat[1 + currentIndex][1] -= 0;
                        //mat[2 + currentIndex][1] -= 0;
                    }
                    else {
                        intermediatestep = (i * x + j * y + k * z) / (i * i + j * j + k * k);
                        mat[0 + currentIndex][0] -= i * intermediatestep;
                        mat[1 + currentIndex][0] -= j * intermediatestep;
                        mat[2 + currentIndex][0] -= k * intermediatestep;

                        //imaginary part
                        x = mat[0 + currentIndex][1];
                        y = mat[1 + currentIndex][1];
                        z = mat[2 + currentIndex][1];
                        intermediatestep = (i * x + j * y + k * z) / (i * i + j * j + k * k);
                        mat[0 + currentIndex][1] -= i * intermediatestep;
                        mat[1 + currentIndex][1] -= j * intermediatestep;
                        mat[2 + currentIndex][1] -= k * intermediatestep;
                    }
                }
            }
            
           
        }



        fftw_execute(pbwdvec);
    }
    void CalcMONDPot(FixedSizeMat2 OldPot, fftw_plan pfwdvec, fftw_plan pbwd) {

        fftw_execute(pfwdvec);
        for (int i = -halfpixels; i < halfpixels; ++i) {
            for (int j = -halfpixels; j < halfpixels; ++j) {
                for (int k = -halfpixels; k < halfpixels; ++k) {

                    double x, y, z, intermediatestep;
                    int currentIndex =  ((k + 2 * halfpixels) % (2 * halfpixels) + 2 * halfpixels * ((j + 2 * halfpixels) % (2 * halfpixels) + 2 * halfpixels * ((i + 2 * halfpixels) % (2 * halfpixels))));
                    //real part;


                    if (i == 0 and j == 0 and k == 0) {
                        OldPot.mat[currentIndex][0] =0 ;
                        OldPot.mat[currentIndex][1] = 0;
                        // mat[0 + currentIndex][0] -= 0;
                        // mat[1 + currentIndex][0] -= 0;
                        //mat[2 + currentIndex][0] -= 0;

                        // mat[0 + currentIndex][1] -= 0;
                        // mat[1 + currentIndex][1] -= 0;
                         //mat[2 + currentIndex][1] -= 0;
                    }
                    else {


                        x = mat[0 + 3*currentIndex][0];
                        y = mat[1 + 3*currentIndex][0];
                        z = mat[2 + 3*currentIndex][0];
                        intermediatestep = (i * x + j * y + k * z) / (i * i + j * j + k * k);
                        OldPot.mat[currentIndex][0] = -1 * intermediatestep / kstep;


                        x = mat[0 + 3*currentIndex][1];
                        y = mat[1 + 3*currentIndex][1];
                        z = mat[2 + 3*currentIndex][1];
                        intermediatestep = (i * x + j * y + k * z) / (i * i + j * j + k * k);
                        OldPot.mat[currentIndex][1] = -1*intermediatestep / kstep;
                        
                    }
                }
            }
        }

        fftw_execute(pbwd);

        for (int i = 0; i < 2 * halfpixels; ++i) { //probably not efficient
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    OldPot.mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)][0] = OldPot.mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)][1];
                    OldPot.mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)][1] = 0;

                }
            }
        }


    }
    FixedSizeMatVec MulOutPlace(FixedSizeMat2 A) {
        FixedSizeMatVec res;
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        res.mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0] = mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0]*A.mat[(k + 2 * halfpixels * (j + 2 * halfpixels * i))][0];

                    }
                }
            }

        }
        return res;
    }
    void operator-(FixedSizeMatVec A) {
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    for (int l = 0; l < 3; ++l) {

                        mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0] = mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0]- A.mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][0];

                    }
                }
            }
        }
    }
    void real() {
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    for (int l = 0; l < 3; ++l) {

                        mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))][1] = 0;

                    }
                }
            }
        }
    }
};

class TwoVecMats {
public:
    FixedSizeMatVec gM2;
    FixedSizeMatVec H;


};


class Particlelist{
    
    public:
        Eigen::Matrix<float, Eigen::Dynamic, 7> plist;
        float EPot;
        Particlelist() {
            EPot = 0;
        }
        Particlelist(Eigen::Matrix<float, N, 7> particlelist){
            plist=particlelist;
            EPot = 0;
        }

        float Ekin(){
            return 0.5*plist.col(0).dot((plist.block(0,4,plist.rows(),3)*plist.block(0,4,plist.rows(),3).transpose()).diagonal());
        }

        float AngMom(){
            Eigen::Matrix<float,Eigen::Dynamic,3> crossMat(plist.rows(),3);
            for(int i=0;i<plist.rows();++i){

                crossMat(i,0)=plist(i,2)*plist(i,6)-plist(i,3)*plist(i,5);
                crossMat(i,1)=plist(i,3)*plist(i,4)-plist(i,1)*plist(i,6);
                crossMat(i,2)=plist(i,1)*plist(i,5)-plist(i,2)*plist(i,4);
                
            }
            return (plist.block(0,0,plist.rows(),1).asDiagonal()*crossMat).sum();
        }

        Eigen::Matrix<float, Eigen::Dynamic, 3> UpdateAccsMOND(int M = 4, float sigma = 1, int iterlen = 4, int regime = 0,fftw_complex* in=NULL, fftw_complex* invecH=NULL,fftw_plan pfwd=NULL,fftw_plan pbwd=NULL,fftw_plan pfwdvecH=NULL,fftw_plan pbwdvecH=NULL) {
            FixedSizeMat2 density;
            FixedSizeMat2 densitycopy;
           // FixedSizeMat2 Pot;
            Eigen::Matrix<float, N, 3> accparts = Eigen::Matrix<float, 2, 3>::Zero();
            TwoVecMats HNDacc;
            
            //Pot = density;
            fftw_free(density.mat);
            fftw_free(HNDacc.H.mat);
            density.mat = in;
            HNDacc.H.mat = invecH;
            density.fillzero();

            density = AssignMassGauss(density, *this, 4);
            densitycopy = density.copy(densitycopy);
           // Pot = CalcPot(density, pfwd, pbwd);
            //HNDacc.gM2 = CalcAcc(Pot, HNDacc.gM2);
            density = CalcPot(density, pfwd, pbwd);
            HNDacc.gM2 = CalcAcc(density, HNDacc.gM2);


            int iters = 4;
            for (int i = 0; i < iters; ++i) {
                HNDacc = MainLoop(HNDacc, 0, pfwdvecH, pbwdvecH, iters - i);
            }

            HNDacc.gM2.CalcMONDPot(density, pfwdvecH, pbwd); //This only changes the density matrix
            density* (1 / (std::pow(2 * halfpixels, 3)));

            densitycopy* density;
            EPot = densitycopy.sum() * cellvolume;
            fftw_free(densitycopy.mat);

            HNDacc.gM2 = CalcAcc(density, HNDacc.gM2);
            accparts = AssignAccsGauss(HNDacc.gM2, *this, accparts, 4);

            return accparts;


        }
        void TimeSim( float dt, int iterlength, int regime = 0,fftw_complex * in = NULL, fftw_complex * invecH = NULL, fftw_plan pfwd = NULL, fftw_plan pbwd = NULL, fftw_plan pfwdvecH = NULL, fftw_plan pbwdvecH = NULL) {
            
            float* posmat=(float*) std::malloc(T * 3*N*sizeof(float));
            float* vecmat=(float*) std::malloc(T * 3 * N * sizeof(float));
            float* MomMat = (float*)std::malloc(T * 3 * sizeof(float));
            float* AngMat=(float*)std::malloc(T * 3  * sizeof(float));
            float* EkinMat = (float*)std::malloc(T * sizeof(float));
            float* EPotMat = (float*)std::malloc(T *  sizeof(float));

            Eigen::Matrix<float, N, 3> accnew = Eigen::Matrix<float, 2, 3>::Zero();
            Eigen::Matrix<float, N, 3> accold = Eigen::Matrix<float, 2, 3>::Zero();

            accnew = this->UpdateAccsMOND(4, 1, 4, regime, in,invecH,pfwd,pbwd,pfwdvecH,pbwdvecH);

            for (int t = 0; t < T; ++t) {
                for (int n = 0; n < N; ++n) {
                    posmat[0 + 3 * (n + N * t)] = plist(n, 1);
                    posmat[1 + 3 * (n + N * t)] = plist(n, 2);
                    posmat[2 + 3 * (n + N * t)] = plist(n, 3);
                    vecmat[0 + 3 * (n + N * t)] = plist(n, 4);
                    vecmat[1 + 3 * (n + N * t)] = plist(n, 5);
                    vecmat[2 + 3 * (n + N * t)] = plist(n, 6);
                }
                //MomMat
               // AngMat[]
                EkinMat[t] = this->Ekin();
                EPotMat[t] = EPot;
            
            accold = accnew;
            plist.block(0, 1, N, 3) += plist.block(0, 4, N, 3) * dt + 0.5 * cellleninv * std::pow(dt, 2) * accold.block(0, 0, N, 3);
            try {
                accnew= this->UpdateAccsMOND(4, 1, 4, regime, in, invecH, pfwd, pbwd, pfwdvecH, pbwdvecH);
            }
            catch(...) {
                break;
            }
            plist.block(0, 4, N, 3) += (accold + accnew) * 0.5 * dt * cellleninv;

            }
            write_csv("posmat.csv", posmat, T * 3 * N);
            write_csv("vecmat.csv", vecmat, T * 3 * N);
            write_csv("Ekinmat.csv", EkinMat, T);
            write_csv("EPotmat.csv", EPotMat, T);
        }

};
class TwoBodyParticlelist : public Particlelist {

    public:
        float m1;
        float m2;
        TwoBodyParticlelist(float mass1, float mass2, Eigen::Vector<float,3> rvec1, Eigen::Vector<float, 3> rvec2, Eigen::Vector<float, 3> vvec1, Eigen::Vector<float, 3> vvec2 ) {
            Eigen::Vector<float, 3> mid;
            mid<< halfpixels / 2, halfpixels / 2, halfpixels / 2;
            rvec1 += mid;
            rvec2 += mid;
            m1 = mass1;
            m2 = mass2;
            plist(0, 0) = m1; plist(1, 0) = m2;
            plist.block(0, 1, 0, 3) = rvec1; plist.block(1, 1, 0, 3) = rvec2;
            plist.block(0, 4, 0, 3) = vvec1; plist.block(1, 4, 0, 3) = vvec2;
        }
        Eigen::Vector<float, 3> AnalyticalForce() {

        }
        float EPotAna() {

        }
};

/*class FixedSizeMat {
    public:
        std::array<std::array<std::array<float,2*halfpixels>,2*halfpixels>,2*halfpixels>* mat=(std::array<std::array<std::array<float, 2 * halfpixels>, 2 * halfpixels>, 2 * halfpixels>*) malloc(sizeof(std::array<std::array<std::array<float, 2 * halfpixels>, 2 * halfpixels>, 2 * halfpixels>));
        FixedSizeMat(){
        for(int i=0;i<2*halfpixels;++i){
            for(int j=0;j<2*halfpixels;++j){
                for(int k=0;k<2*halfpixels;++k){
                    *mat[i][j][k]=0;
                }
            }
        }

        }
        FixedSizeMat operator*(float a){
            FixedSizeMat fixMat;
        for(int i=0;i<2*halfpixels;++i){
            for(int j=0;j<2*halfpixels;++j){
                for(int k=0;k<2*halfpixels;++k){
                    fixMat.mat[i][j][k]=a* *mat[i][j][k];
                }
            }
        }
        return fixMat;
        }
        void FFT(){

        }
};
*/


float EuclidDist(Eigen::Vector3f a, Eigen::Vector3f b) {
    int two = 2;
    return std::sqrt(std::pow(a(0) - b(0), two) + std::pow(a(1) - b(1), two) + std::pow(a(2) - b(2), two));
}



FixedSizeMat2 AssignMassGauss(FixedSizeMat2 density, Particlelist particlelist, int N2, float sigma ) {





    for (int i = 0; i < N; ++i) {
        float x = particlelist.plist(i, 1);
        float y = particlelist.plist(i, 2);
        float z = particlelist.plist(i, 3);

        for (int a = -N2 + 1; a < N2 + 1; ++a) {
            for (int b = -N2 + 1; b < N2 + 1; ++b) {
                for (int c = -N2 + 1; c < N2 + 1; ++c) {
                    std::array<int, 3> cellcoords;
                    cellcoords[0] = a + (int)x;
                    cellcoords[1] = b + (int)y;
                    cellcoords[2] = c + (int)z;
                    float weight = oversqrt2pi3 * std::exp(-(std::pow(cellcoords[0] - x, 2) + std::pow(cellcoords[1] - y, 2) + std::pow(cellcoords[2] - z, 2)) / (2.0 * std::pow(sigma, 2)));
                    *density(cellcoords[0], cellcoords[1], cellcoords[2], 'r') += particlelist.plist(i, 0) * weight;


                }
            }
        }
    }

    float sigma3inv = std::pow(sigma, -3);

    density* (cellvolumeinv * sigma3inv);
    return density;
}
Eigen::Matrix<float, Eigen::Dynamic, 3> AssignAccsGauss(FixedSizeMatVec accmat, Particlelist particlelist, Eigen::Matrix<float, Eigen::Dynamic, 3> accparts, int N, float sigma ) {


    float sigma3inv = std::pow(sigma, -3);


    for (int i = 0; i < particlelist.plist.rows(); ++i) {
        float x = particlelist.plist(i, 1);
        float y = particlelist.plist(i, 2);
        float z = particlelist.plist(i, 3);

        for (int a = -N + 1; a < N + 1; ++a) {
            for (int b = -N + 1; b < N + 1; ++b) {
                for (int c = -N + 1; c < N + 1; ++c) {
                    std::array<int, 3> cellcoords;
                    cellcoords[0] = a + (int)x;
                    cellcoords[1] = b + (int)y;
                    cellcoords[2] = c + (int)z;
                    float weight = oversqrt2pi3 * std::exp(-(std::pow(cellcoords[0] - x, 2) + std::pow(cellcoords[1] - y, 2) + std::pow(cellcoords[2] - z, 2)) / (2.0 * std::pow(sigma, 2)));

                    accparts(i, 0) += *accmat(cellcoords[0], cellcoords[1], cellcoords[2], 0) * weight * sigma3inv;
                    accparts(i, 1) += *accmat(cellcoords[0], cellcoords[1], cellcoords[2], 1) * weight * sigma3inv;
                    accparts(i, 2) += *accmat(cellcoords[0], cellcoords[1], cellcoords[2], 2) * weight * sigma3inv;


                }
            }
        }
    }




    return accparts;
}

FixedSizeMat2 CalcPot(FixedSizeMat2 Density, fftw_plan pfwd, fftw_plan pbwd) {
    fftw_execute(pfwd);

    Density* (-c / std::pow(kstep, 2));

    for (int i = -halfpixels; i < halfpixels; ++i) {
        for (int j = -halfpixels; j < halfpixels; ++j) {
            for (int k = -halfpixels; k < halfpixels; ++k) {
                int KLM = i * i + j * j + k * k;
                float KLMinv;
                if (KLM != 0) {
                    KLMinv = 1.0 / KLM;
                }
                else {
                    KLMinv = 1;
                }


                *Density((i + 2 * halfpixels) % (2 * halfpixels), (j + 2 * halfpixels) % (2 * halfpixels), (k + 2 * halfpixels) % (2 * halfpixels), 'r') *= KLMinv;
                *Density((i + 2 * halfpixels) % (2 * halfpixels), (j + 2 * halfpixels) % (2 * halfpixels), (k + 2 * halfpixels) % (2 * halfpixels), 'i') *= KLMinv;

            }
        }
    }

    fftw_execute(pbwd);
    Density* (1 / (std::pow(2 * halfpixels, 3)));
    return Density;
}

FixedSizeMatVec CalcAcc(FixedSizeMat2 Pot, FixedSizeMatVec accmat) {

    for (int i = 0; i < 2 * halfpixels; ++i) {
        for (int j = 0; j < 2 * halfpixels; ++j) {
            for (int k = 0; k < 2 * halfpixels; ++k) {

                *accmat(i, j, k, 0) = -(*Pot((2 * halfpixels + ((i + 1) % (2 * halfpixels))) % (2 * halfpixels), j, k, 'r') - *Pot((2 * halfpixels + ((i - 1) % (2 * halfpixels))) % (2 * halfpixels), j, k, 'r')) / (2 * celllen);
                *accmat(i, j, k, 1) = -(*Pot(i, (2 * halfpixels + ((j + 1) % (2 * halfpixels))) % (2 * halfpixels), k, 'r') - *Pot(i, (2 * halfpixels + ((j - 1) % (2 * halfpixels))) % (2 * halfpixels), k, 'r')) / (2 * celllen);
                *accmat(i, j, k, 2) = -(*Pot(i, j, (2 * halfpixels + ((k + 1) % (2 * halfpixels))) % (2 * halfpixels), 'r') - *Pot(i, j, (2 * halfpixels + ((k - 1) % (2 * halfpixels))) % (2 * halfpixels), 'r')) / (2 * celllen);
            }
        }
    }
    return accmat;
}

FixedSizeMat2 inpol(FixedSizeMatVec x, int func) {
    FixedSizeMat2 xnorm = x.norm();
    xnorm* (1 / a0);
    if (func == 0) {
        return xnorm;
    }
    //if (func == 5) {
       // return 1
    //}
}

FixedSizeMat2 inpolinv(FixedSizeMatVec x, int func) {
    FixedSizeMat2 xnorm = x.norm();

    xnorm* (1 / a0);
    if (func == 0) {
        xnorm.sqrtinv();
        return xnorm;
    }
}

TwoVecMats MainLoop(TwoVecMats HNDacc, int func, fftw_plan pfwdvecH, fftw_plan pbwdvecH, int lastit) {
    FixedSizeMat2 inpolres;
    fftw_free(inpolres.mat);

    HNDacc.H + HNDacc.gM2;
    inpolres = inpolinv(HNDacc.H, func);
    HNDacc.H* inpolres;
    fftw_free(inpolres.mat);
    HNDacc.H.CurlFreeProj(pfwdvecH, pbwdvecH);
    HNDacc.H.real();
    HNDacc.H* (1 / (std::pow(2 * halfpixels, 3)));

    if (lastit == 1) {
        fftw_free(HNDacc.gM2.mat);
        HNDacc.gM2.mat = HNDacc.H.mat;
        return HNDacc;
    }
    inpolres = (inpol(HNDacc.H, func));
    HNDacc.H* inpolres;
    fftw_free(inpolres.mat);
    HNDacc.H - HNDacc.gM2; // this subtraction is in place and changes H
    HNDacc.H.DivFreeProj(pfwdvecH, pbwdvecH);
    HNDacc.H.real();
    HNDacc.H* (1 / (std::pow(2 * halfpixels, 3)));

    return HNDacc;

}





int main(){
    float v = 2 * 5.567090223714246e-06;
    Eigen::Matrix<float, 2, 7> pmat;
    pmat << std::pow(10,20), 32.1, 36.7, 32.3, 0, 1.5/2.5*v, 0,
          1.5*std::pow(10,20), 34.1, 36.7, 32.3, 0, -1/2.5*v, 0;
    Particlelist plist(pmat);

 

    fftw_complex* in;
    fftw_complex* invecH;
    //fftw_complex* invecgM2;
    in = (fftw_complex*)fftw_malloc(2 * halfpixels * 2 * halfpixels * 2 * halfpixels * sizeof(fftw_complex));
    invecH = (fftw_complex*)fftw_malloc(3*2 * halfpixels * 2 * halfpixels * 2 * halfpixels * sizeof(fftw_complex));
    //invecgM2 = (fftw_complex*)fftw_malloc(3 * 2 * halfpixels * 2 * halfpixels * 2 * halfpixels * sizeof(fftw_complex));

    fftw_plan pfwd;
    fftw_plan pbwd;
    fftw_plan pfwdvecH;
    fftw_plan pbwdvecH;


    int size[] = { 2 * halfpixels,2 * halfpixels,2 * halfpixels };

    pfwd = fftw_plan_dft_3d(2 * halfpixels, 2 * halfpixels, 2 * halfpixels, in, in, FFTW_FORWARD, FFTW_MEASURE);
    pbwd= fftw_plan_dft_3d(2 * halfpixels, 2 * halfpixels, 2 * halfpixels, in, in, FFTW_BACKWARD, FFTW_MEASURE);

    pfwdvecH = fftw_plan_many_dft(3, size, 3, invecH,NULL,3,1, invecH, NULL, 3, 1, FFTW_FORWARD, FFTW_MEASURE);
    pbwdvecH = fftw_plan_many_dft(3, size, 3, invecH, NULL, 3, 1, invecH, NULL, 3, 1, FFTW_BACKWARD, FFTW_MEASURE);


    plist.TimeSim(86400, 4, 0, in, invecH, pfwd, pbwd, pfwdvecH, pbwdvecH);
    return 0;
}
