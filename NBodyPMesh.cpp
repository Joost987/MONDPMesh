
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

const int halfpixels=128;
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
class FixedSizeMatComplex;
class FixedSizeMatVecComplex;


Eigen::Matrix<float, Eigen::Dynamic, 3> AssignAccsGauss(FixedSizeMatVec accmat, Particlelist particlelist, Eigen::Matrix<float, Eigen::Dynamic, 3> accparts, int N, float sigma=1);
FixedSizeMat2 CalcPot(FixedSizeMat2 Density,FixedSizeMatComplex Densityfft, fftwf_plan pfwd, fftwf_plan pbwd); 
FixedSizeMatVec CalcAcc(FixedSizeMat2 Pot, FixedSizeMatVec accmat); 
FixedSizeMat2 inpol(FixedSizeMatVec x, int func); 
FixedSizeMat2 inpolinv(FixedSizeMatVec x, int func); 
TwoVecMats MainLoop(TwoVecMats HNDacc, FixedSizeMatVecComplex Hfft, int func, fftwf_plan pfwdvecH, fftwf_plan pbwdvecH, int lastit);
FixedSizeMat2 AssignMassGauss(FixedSizeMat2 density, Particlelist particlelist, int N, float sigma=1);

void write_csv(std::string filename, float* mat, int size) {
    std::ofstream myFile(filename);
    for (int i = 0; i < size; ++i) {
        myFile << *(mat + i)<<";";

    }
    myFile.close();
}
class FixedSizeMatComplex {
public:
    fftwf_complex* mat;
    FixedSizeMatComplex(){
        mat= (fftwf_complex*)fftwf_malloc(2 * halfpixels * 2 * halfpixels * (halfpixels + 1) * sizeof(fftwf_complex));
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < halfpixels+1; ++k) {
                    mat[k +  (halfpixels+1) * (j + 2 * halfpixels * i)][0] = 0;
                    mat[k + (halfpixels + 1) * (j + 2 * halfpixels * i)][1] = 0;

                }
            }
        }
    }
    float* operator()(int i, int j, int k, char t) {
    float *val=NULL;

    if (t == 'r') {
        val = &mat[k + (halfpixels + 1) * (j + 2 * halfpixels * i)][0];
    }
    if (t == 'i') {
       val = &mat[k + (halfpixels + 1) * (j + 2 * halfpixels * i)][1];
    }

    return val;
}
    void operator*(float a) {
        //Note: This multiplication is in place
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                 for (int k = 0; k < halfpixels+1; ++k) {
                    mat[k + (halfpixels + 1) * (j + 2 * halfpixels * i)][0] *= a;
                    mat[k + (halfpixels + 1) * (j + 2 * halfpixels * i)][1] *= a;
                }
            }
        }

    }
};

class FixedSizeMatVecComplex {
public:
    fftwf_complex* mat;
    FixedSizeMatVecComplex() {
        mat = (fftwf_complex*)fftwf_malloc(3 *2* halfpixels * 2 * halfpixels *  (halfpixels+1) * sizeof(fftwf_complex));
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        mat[l + 3 * (k + (halfpixels + 1) * (j + 2 * halfpixels * i))][0] = 0;
                        mat[l + 3 * (k + (halfpixels + 1) * (j + 2 * halfpixels * i))][1] = 0;
                    }
                }
            }
        }
    }
};
class FixedSizeMat2 {
    public:
        float* mat;
        FixedSizeMat2() {
            mat = fftwf_alloc_real(2 * halfpixels * 2 * halfpixels * 2 * (halfpixels) * sizeof(float));
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                        mat[k+2*halfpixels*(j+2*halfpixels*i)] = 0;

                    }
                }
            }
           


        }
        
        FixedSizeMat2 copy(FixedSizeMat2 A) {
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                        A.mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)] = mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)];
                    }
                }
            }
            return A;
        }

        float* operator()(int i, int j, int k) {
            float *val=NULL;
            val = &mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)];
            return val;
        }

        void operator*(float a) {
            //Note: This multiplication is in place
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                        mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)] *= a;
                    }
                }
            }

        }
        void operator*(FixedSizeMat2 A) {
            //Note: This multiplication is in place
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                        mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)] *= A.mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)];


                    }
                }
            }

        }
        float sum() {
            float res = 0;
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                        res += mat[k + 2 * halfpixels * (j + 2 * halfpixels * i)];


                    }
                }
            }
            return res;
        }
        void sqrtinv() {
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                            mat[ (k + 2 * halfpixels * (j + 2 * halfpixels * i))] = 1.0 / std::sqrt(mat[ (k + 2 * halfpixels * (j + 2 * halfpixels * i))]);
                    }
                }
            }
        }
        void fillzero() {
            for (int i = 0; i < 2 * halfpixels; ++i) {
                for (int j = 0; j < 2 * halfpixels; ++j) {
                    for (int k = 0; k < 2 * halfpixels; ++k) {
                        mat[(k + 2 * halfpixels * (j + 2 * halfpixels * i))] = 0;
                    }
                }
            }
        }
};

class FixedSizeMatVec {
public:
    float* mat;
    FixedSizeMatVec() {
        mat= fftwf_alloc_real(3*2 * halfpixels * 2 * halfpixels * 2 * (halfpixels) * sizeof(float));
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        mat[l+3*(k + 2 * halfpixels * (j + 2 * halfpixels * i))] = 0;
                    }
                }
            }
        }
    }
    
    float* operator()(int i, int j, int k, int l) {
        float* val = NULL;
        val = &mat[l+3*( k + 2 * halfpixels * (j + 2 * halfpixels * i))];
        return val;
    }

    void operator*(float a) {
        //Note: This multiplication is in place
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))] *= a;
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
                        mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))] *= A.mat[(k + 2 * halfpixels * (j + 2 * halfpixels * i))];

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
                        mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))] += A.mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))];
                    }
                }
            }
        }
    }
    void operator-(FixedSizeMatVec A) {
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))] = mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))] - A.mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))];

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
                        mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))] = 1.0/std::sqrt(mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))]);
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
                    float x, y, z;
                    x = mat[0 + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))];
                    y = mat[1 + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))];
                    z = mat[2 + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))];
                    *normmat(i, j, k) = std::sqrt(x * x + y * y + z * z);
                    
                }
            }
        }
        return normmat;
    }
    void CurlFreeProj(FixedSizeMatVecComplex matfft, fftwf_plan pfwdvec, fftwf_plan pbwdvec) {
       
        fftwf_execute(pfwdvec);

        for (int i = -halfpixels + 1; i < halfpixels + 1; ++i) {
            for (int j = -halfpixels + 1; j < halfpixels + 1; ++j) {
                for (int k = 0; k < halfpixels + 1; ++k) {
                    float x, y, z, intermediatestep;
                    int currentIndex = 3 * ((k + 2 * halfpixels) % (2 * halfpixels) + ( halfpixels+1) * ((j + 2 * halfpixels) % (2 * halfpixels) + 2 * halfpixels * ((i + 2 * halfpixels) % (2 * halfpixels))));
                    //real part;
                    
                    x = matfft.mat[0 + currentIndex][0];
                    y = matfft.mat[1 + currentIndex][0];
                    z = matfft.mat[2 + currentIndex][0];
                    if(i==0 and j==0 and k==0){
                        matfft.mat[0 + currentIndex][0] = 0;
                        matfft.mat[1 + currentIndex][0] = 0;
                        matfft.mat[2 + currentIndex][0] = 0;

                        matfft.mat[0 + currentIndex][1] = 0;
                        matfft.mat[1 + currentIndex][1] = 0;
                        matfft.mat[2 + currentIndex][1] = 0;
                    }
                    else {
                        intermediatestep = (i * x + j * y + k * z) / (i * i + j * j + k * k);
                        matfft.mat[0 + currentIndex][0] = i * intermediatestep;
                        matfft.mat[1 + currentIndex][0] = j * intermediatestep;
                        matfft.mat[2 + currentIndex][0] = k * intermediatestep;

                        //imaginary part
                        x = matfft.mat[0 + currentIndex][1];
                        y = matfft.mat[1 + currentIndex][1];
                        z = matfft.mat[2 + currentIndex][1];
                        intermediatestep = (i * x + j * y + k * z) / (i * i + j * j + k * k);
                        matfft.mat[0 + currentIndex][1] = i * intermediatestep;
                        matfft.mat[1 + currentIndex][1] = j * intermediatestep;
                        matfft.mat[2 + currentIndex][1] = k * intermediatestep;
                    }
                }
            }
        }

        fftwf_execute(pbwdvec);
        
    }

    void DivFreeProj(FixedSizeMatVecComplex matfft, fftwf_plan pfwdvec, fftwf_plan pbwdvec) {

        fftwf_execute(pfwdvec);

        for (int i = -halfpixels + 1; i < halfpixels + 1; ++i) {
            for (int j = -halfpixels + 1; j < halfpixels + 1; ++j) {
                for (int k = 0; k < halfpixels + 1; ++k) {
                    float x, y, z, intermediatestep;
                    int currentIndex = 3 * ((k + 2 * halfpixels) % (2 * halfpixels) + (halfpixels+1) * ((j + 2 * halfpixels) % (2 * halfpixels) + 2 * halfpixels * ((i + 2 * halfpixels) % (2 * halfpixels))));
                    //real part;
                    
                    x = matfft.mat[0 + currentIndex][0];
                    y = matfft.mat[1 + currentIndex][0];
                    z = matfft.mat[2 + currentIndex][0];
                    if (i == 0 and j == 0 and k == 0) {

                    }
                    else {
                        intermediatestep = (i * x + j * y + k * z) / (i * i + j * j + k * k);
                        matfft.mat[0 + currentIndex][0] -= i * intermediatestep;
                        matfft.mat[1 + currentIndex][0] -= j * intermediatestep;
                        matfft.mat[2 + currentIndex][0] -= k * intermediatestep;

                        //imaginary part
                        x = matfft.mat[0 + currentIndex][1];
                        y = matfft.mat[1 + currentIndex][1];
                        z = matfft.mat[2 + currentIndex][1];
                        intermediatestep = (i * x + j * y + k * z) / (i * i + j * j + k * k);
                        matfft.mat[0 + currentIndex][1] -= i * intermediatestep;
                        matfft.mat[1 + currentIndex][1] -= j * intermediatestep;
                        matfft.mat[2 + currentIndex][1] -= k * intermediatestep;
                    }
                }
            }
            
           
        }



        fftwf_execute(pbwdvec);
    }
    void CalcMONDPot(FixedSizeMat2 OldPot, FixedSizeMatComplex OldPotfft, FixedSizeMatVecComplex matfft,fftwf_plan pfwdvec, fftwf_plan pbwd) {

        fftwf_execute(pfwdvec);

        for (int i = -halfpixels + 1; i < halfpixels + 1; ++i) {
            for (int j = -halfpixels + 1; j < halfpixels + 1; ++j) {
                for (int k = 0; k < halfpixels + 1; ++k) {
                    float x, y, z, intermediatestep;
                    int currentIndex = ((k + 2 * halfpixels) % (2 * halfpixels) + (halfpixels+1) * ((j + 2 * halfpixels) % (2 * halfpixels) + 2 * halfpixels * ((i + 2 * halfpixels) % (2 * halfpixels))));

                    if (i == 0 and j == 0 and k == 0) {
                        OldPotfft.mat[currentIndex][0] =0 ;
                        OldPotfft.mat[currentIndex][1] = 0;
                    }
                    else {
                        x = matfft.mat[0 + 3*currentIndex][0];
                        y = matfft.mat[1 + 3*currentIndex][0];
                        z = matfft.mat[2 + 3*currentIndex][0];
                        intermediatestep = (i * x + j * y + k * z) / (i * i + j * j + k * k);
                        OldPotfft.mat[currentIndex][0] = -1 * intermediatestep / kstep; //real part


                        x = matfft.mat[0 + 3*currentIndex][1];
                        y = matfft.mat[1 + 3*currentIndex][1];
                        z = matfft.mat[2 + 3*currentIndex][1];
                        intermediatestep = (i * x + j * y + k * z) / (i * i + j * j + k * k);
                        OldPotfft.mat[currentIndex][1] = -1*intermediatestep / kstep; //imaginary part
                        
                    }
                }
            }
        }//Multiply by -i, since we need the - imaginary part of the transform
        

        float temp;
        for (int i = 0; i < halfpixels*2; ++i) {
            for (int j = 0; j < halfpixels *2; ++j) {
                for (int k = 0; k < halfpixels + 1; ++k) {
                    temp = OldPotfft.mat[k + (halfpixels + 1) * (j + 2 * halfpixels * i)][0];
                    OldPotfft.mat[k + (halfpixels + 1) * (j + 2 * halfpixels * i)][0] = OldPotfft.mat[k + (halfpixels + 1) * (j + 2 * halfpixels * i)][1];
                    OldPotfft.mat[k + (halfpixels + 1) * (j + 2 * halfpixels * i)][1] = -1 * temp;
                }
            }
        }
        fftwf_execute(pbwd);





    }
    FixedSizeMatVec MulOutPlace(FixedSizeMat2 A) {
        FixedSizeMatVec res;
        for (int i = 0; i < 2 * halfpixels; ++i) {
            for (int j = 0; j < 2 * halfpixels; ++j) {
                for (int k = 0; k < 2 * halfpixels; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        res.mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))] = mat[l + 3 * (k + 2 * halfpixels * (j + 2 * halfpixels * i))]*A.mat[(k + 2 * halfpixels * (j + 2 * halfpixels * i))];

                    }
                }
            }

        }
        return res;
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

        Eigen::Matrix<float, Eigen::Dynamic, 3> UpdateAccsMOND(int M = 4, float sigma = 1, int iterlen = 4, int regime = 0,float* in=NULL, float* invecH=NULL,fftwf_complex* out=NULL,fftwf_complex* outvecH=NULL,fftwf_plan pfwd=NULL,fftwf_plan pbwd=NULL,fftwf_plan pfwdvecH=NULL,fftwf_plan pbwdvecH=NULL) {
            FixedSizeMat2 density;
            FixedSizeMat2 densitycopy;
            FixedSizeMatComplex densityfft;
            FixedSizeMatVecComplex Hfft;
            Eigen::Matrix<float, N, 3> accparts = Eigen::Matrix<float, N, 3>::Zero();
            TwoVecMats HNDacc;
            
            fftwf_free(density.mat);
            fftwf_free(HNDacc.H.mat);
            fftwf_free(densityfft.mat);
            fftwf_free(Hfft.mat);
            density.mat = in;
            HNDacc.H.mat = invecH;
            densityfft.mat = out;
            Hfft.mat = outvecH;
            density.fillzero();

            density = AssignMassGauss(density, *this, 4);
            densitycopy = density.copy(densitycopy);
            density = CalcPot(density,densityfft, pfwd, pbwd);
            HNDacc.gM2 = CalcAcc(density, HNDacc.gM2);



            int iters = 4;
            for (int i = 0; i < iters; ++i) {
                HNDacc = MainLoop(HNDacc,Hfft, 0, pfwdvecH, pbwdvecH, iters - i);
            }

            HNDacc.gM2.CalcMONDPot(density, densityfft,Hfft,pfwdvecH, pbwd); //This only changes the density matrix
            density* (1 / (std::pow(2 * halfpixels, 3)));
            densitycopy* density;
            EPot = densitycopy.sum() * cellvolume;
            fftwf_free(densitycopy.mat);

            HNDacc.gM2 = CalcAcc(density, HNDacc.gM2);
            accparts = AssignAccsGauss(HNDacc.gM2, *this, accparts, 4);

            return accparts;


        }
        void TimeSim( float dt, int iterlength, int regime = 0,float * in = NULL, float * invecH = NULL, fftwf_complex* out = NULL, fftwf_complex* outvecH = NULL, fftwf_plan pfwd = NULL, fftwf_plan pbwd = NULL, fftwf_plan pfwdvecH = NULL, fftwf_plan pbwdvecH = NULL) {
            
            float* posmat=(float*) std::malloc(T * 3*N*sizeof(float));
            float* vecmat=(float*) std::malloc(T * 3 * N * sizeof(float));
            float* MomMat = (float*)std::malloc(T * 3 * sizeof(float));
            float* AngMat=(float*)std::malloc(T * 3  * sizeof(float));
            float* EkinMat = (float*)std::malloc(T * sizeof(float));
            float* EPotMat = (float*)std::malloc(T *  sizeof(float));

            Eigen::Matrix<float, N, 3> accnew = Eigen::Matrix<float, N, 3>::Zero();
            Eigen::Matrix<float, N, 3> accold = Eigen::Matrix<float, N, 3>::Zero();

            accnew = this->UpdateAccsMOND(4, 1, 4, regime, in,invecH,out,outvecH,pfwd,pbwd,pfwdvecH,pbwdvecH);

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
                accnew= this->UpdateAccsMOND(4, 1, 4, regime, in, invecH,out,outvecH, pfwd, pbwd, pfwdvecH, pbwdvecH);
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
                    *density(cellcoords[0], cellcoords[1], cellcoords[2]) += particlelist.plist(i, 0) * weight;


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

FixedSizeMat2 CalcPot(FixedSizeMat2 Density, FixedSizeMatComplex Densityfft, fftwf_plan pfwd, fftwf_plan pbwd) {

    fftwf_execute(pfwd);
    Densityfft* (-c / std::pow(kstep, 2));

   for (int i = -halfpixels+1; i < halfpixels+1; ++i) {
         for (int j = -halfpixels+1; j < halfpixels+1; ++j) {
            for (int k = 0; k < halfpixels+1; ++k) {
                int KLM = i * i + j * j + k * k;
                float KLMinv;
                if (KLM != 0) {
                    KLMinv = 1.0 / KLM;
                }
                else {
                    KLMinv = 1;
                }

                *Densityfft((i + 2 * halfpixels) % (2 * halfpixels), (j + 2 * halfpixels) % (2 * halfpixels), (k + 2 * halfpixels) % (2 * halfpixels), 'r') *= KLMinv;
                *Densityfft((i + 2 * halfpixels) % (2 * halfpixels), (j + 2 * halfpixels) % (2 * halfpixels), (k + 2 * halfpixels) % (2 * halfpixels), 'i') *= KLMinv;

            }
        }
    }

    fftwf_execute(pbwd);
    Density* (1 / ((std::pow(2 * halfpixels, 3))));
    return Density;
}

FixedSizeMatVec CalcAcc(FixedSizeMat2 Pot, FixedSizeMatVec accmat) {

    for (int i = 0; i < 2 * halfpixels; ++i) {
        for (int j = 0; j < 2 * halfpixels; ++j) {
            for (int k = 0; k < 2 * halfpixels; ++k) {

                *accmat(i, j, k, 0) = -(*Pot((2 * halfpixels + ((i + 1) % (2 * halfpixels))) % (2 * halfpixels), j, k) - *Pot((2 * halfpixels + ((i - 1) % (2 * halfpixels))) % (2 * halfpixels), j, k)) / (2 * celllen);
                *accmat(i, j, k, 1) = -(*Pot(i, (2 * halfpixels + ((j + 1) % (2 * halfpixels))) % (2 * halfpixels), k) - *Pot(i, (2 * halfpixels + ((j - 1) % (2 * halfpixels))) % (2 * halfpixels), k)) / (2 * celllen);
                *accmat(i, j, k, 2) = -(*Pot(i, j, (2 * halfpixels + ((k + 1) % (2 * halfpixels))) % (2 * halfpixels)) - *Pot(i, j, (2 * halfpixels + ((k - 1) % (2 * halfpixels))) % (2 * halfpixels))) / (2 * celllen);
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

TwoVecMats MainLoop(TwoVecMats HNDacc, FixedSizeMatVecComplex Hfft, int func, fftwf_plan pfwdvecH, fftwf_plan pbwdvecH, int lastit) {
    FixedSizeMat2 inpolres;
    fftwf_free(inpolres.mat);

    HNDacc.H + HNDacc.gM2; //this changes H
    inpolres = inpolinv(HNDacc.H, func);
    HNDacc.H* inpolres;

    fftwf_free(inpolres.mat);
    HNDacc.H.CurlFreeProj(Hfft,pfwdvecH, pbwdvecH);
    HNDacc.H* (1 / (std::pow(2 * halfpixels, 3)));


    if (lastit == 1) {
        fftwf_free(HNDacc.gM2.mat);
        HNDacc.gM2.mat = HNDacc.H.mat;
        return HNDacc;
    }
    inpolres = (inpol(HNDacc.H, func));
    HNDacc.H* inpolres;
    fftwf_free(inpolres.mat);
    HNDacc.H - HNDacc.gM2; // this subtraction is in place and changes H

    HNDacc.H.DivFreeProj(Hfft, pfwdvecH, pbwdvecH);
    HNDacc.H* (1 /  (std::pow(2 * halfpixels, 3)));

    return HNDacc;

}





int main() {

    float v = 2 * 5.567090223714246e-06;
    Eigen::Matrix<float, N, 7> pmat;
    pmat << std::pow(10, 21), 32.1, 36.7, 32.3, 0, 1.5 / 2.5 * v, 0,
    1.5 * std::pow(10, 20), 36.1, 36.7, 32.3, 0, -1 / 2.5 * v, 0;
   Particlelist plist(pmat);
    Eigen::Vector3f a;
    Eigen::Vector3f b;
    a << 1, 0, 0;
    b << 0, 2, 0;



    //  std::cout<<EuclidDist(a,b);
     // std::cout<<plist.Ekin()<<" ";
      //std::cout<<plist.AngMom()<<" ";

    float* in;
    fftwf_complex* out;
    float* invecH;
    fftwf_complex* outvecH;

    in = fftwf_alloc_real(2 * halfpixels * 2 * halfpixels * 2 * (halfpixels) * sizeof(float));
    out = (fftwf_complex*)fftwf_malloc(2 * halfpixels * 2 * halfpixels * (halfpixels + 1) * sizeof(fftwf_complex));
    invecH = fftwf_alloc_real(3 * 2 * halfpixels * 2 * halfpixels * 2 * (halfpixels) * sizeof(float));
    outvecH = (fftwf_complex*)fftwf_malloc(3 * 2 * halfpixels * 2 * halfpixels * (halfpixels + 1) * sizeof(fftwf_complex));

    fftwf_plan pfwd;
    fftwf_plan pbwd;
    fftwf_plan pfwdvecH;
    fftwf_plan pbwdvecH;


    int size[] = { 2 * halfpixels,2 * halfpixels,2 * halfpixels };
    std::srand(static_cast<unsigned int>(time(nullptr)));

    pfwd = fftwf_plan_dft_r2c_3d(2 * halfpixels, 2 * halfpixels, 2 * halfpixels, in, out, FFTW_MEASURE);
    pbwd = fftwf_plan_dft_c2r_3d(2 * halfpixels, 2 * halfpixels, 2 * halfpixels, out, in, FFTW_MEASURE);

    pfwdvecH = fftwf_plan_many_dft_r2c(3, size, 3, invecH, NULL, 3, 1, outvecH, NULL, 3, 1, FFTW_MEASURE);
    pbwdvecH = fftwf_plan_many_dft_c2r(3, size, 3, outvecH, NULL, 3, 1, invecH, NULL, 3, 1, FFTW_MEASURE);


    Eigen::Matrix<float, Eigen::Dynamic, 3> accparts;

 




    accparts = plist.UpdateAccsMOND(4, 1, 4, 0, in, invecH, out, outvecH, pfwd, pbwd, pfwdvecH, pbwdvecH);
    std::cout<<accparts;



   // plist.TimeSim(86400, 4, 0, in, invecH, pfwd, pbwd, pfwdvecH, pbwdvecH);
    return 0;
}
