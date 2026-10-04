#include <emscripten.h>
#include <emscripten/html5.h>
#include <GLES2/gl2.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>

struct Mat4 { float m[16]; };
static Mat4 matId(){ Mat4 r; memset(r.m,0,sizeof(r.m)); r.m[0]=r.m[5]=r.m[10]=r.m[15]=1; return r; }
static Mat4 matMul(const Mat4&a,const Mat4&b){ Mat4 r;
    for(int c=0;c<4;c++)for(int row=0;row<4;row++){ float s=0;
        for(int k=0;k<4;k++) s+=a.m[k*4+row]*b.m[c*4+k];
        r.m[c*4+row]=s; } return r; }
static Mat4 matT(float x,float y,float z){ Mat4 r=matId(); r.m[12]=x;r.m[13]=y;r.m[14]=z; return r; }
static Mat4 matS(float x,float y,float z){ Mat4 r=matId(); r.m[0]=x;r.m[5]=y;r.m[10]=z; return r; }
static Mat4 matRX(float a){ Mat4 r=matId(); float c=cosf(a),s=sinf(a); r.m[5]=c;r.m[6]=s;r.m[9]=-s;r.m[10]=c; return r; }
static Mat4 matRY(float a){ Mat4 r=matId(); float c=cosf(a),s=sinf(a); r.m[0]=c;r.m[2]=-s;r.m[8]=s;r.m[10]=c; return r; }
static Mat4 matRZ(float a){ Mat4 r=matId(); float c=cosf(a),s=sinf(a); r.m[0]=c;r.m[1]=s;r.m[4]=-s;r.m[5]=c; return r; }
static Mat4 matPersp(float fovy,float asp,float zn,float zf){
    Mat4 r; memset(r.m,0,sizeof(r.m));
    float t=1.0f/tanf(fovy*0.5f);
    r.m[0]=t/asp; r.m[5]=t;
    r.m[10]=(zf+zn)/(zn-zf); r.m[11]=-1.0f;
    r.m[14]=(2.0f*zf*zn)/(zn-zf);
    return r;
}
static Mat4 matLookAt(float ex,float ey,float ez,float cx,float cy,float cz){
    float fx=cx-ex,fy=cy-ey,fz=cz-ez;
    float fl=sqrtf(fx*fx+fy*fy+fz*fz); fx/=fl;fy/=fl;fz/=fl;
    float sx=-fz, sy=0, sz=fx;
    float sl=sqrtf(sx*sx+sz*sz); sx/=sl; sz/=sl;
    float ux=sy*fz-sz*fy, uy=sz*fx-sx*fz, uz=sx*fy-sy*fx;
    Mat4 r=matId();
    r.m[0]=sx; r.m[4]=sy; r.m[8]=sz;
    r.m[1]=ux; r.m[5]=uy; r.m[9]=uz;
    r.m[2]=-fx; r.m[6]=-fy; r.m[10]=-fz;
    r.m[12]=-(sx*ex+sy*ey+sz*ez);
    r.m[13]=-(ux*ex+uy*ey+uz*ez);
    r.m[14]= (fx*ex+fy*ey+fz*ez);
    return r;
}

static const char* VERT = R"(
attribute vec3 aPos;
attribute vec3 aNormal;
uniform mat4 uMVP;
uniform mat4 uModel;
varying vec3 vN;
varying vec3 vP;
void main(){
    vec4 wp = uModel * vec4(aPos,1.0);
    vP = wp.xyz;
    vN = (uModel * vec4(aNormal,0.0)).xyz;
    gl_Position = uMVP * vec4(aPos,1.0);
}
)";
static const char* FRAG = R"(
precision mediump float;
varying vec3 vN;
varying vec3 vP;
uniform vec3 uColor;
void main(){
    vec3 N=normalize(vN);
    vec3 L=normalize(vec3(0.5,1.0,0.7));
    float d=max(dot(N,L),0.0);
    vec3 V=normalize(-vP);
    vec3 H=normalize(L+V);
    float s=pow(max(dot(N,H),0.0),32.0);
    vec3 col=uColor*(0.25+0.75*d)+vec3(1.0)*s*0.45;
    gl_FragColor=vec4(col,1.0);
}
)";

static const float CUBE[] = {
    -0.5f,-0.5f, 0.5f, 0,0,1,  0.5f,-0.5f, 0.5f, 0,0,1,  0.5f, 0.5f, 0.5f, 0,0,1, -0.5f, 0.5f, 0.5f, 0,0,1,
     0.5f,-0.5f,-0.5f, 0,0,-1,-0.5f,-0.5f,-0.5f, 0,0,-1,-0.5f, 0.5f,-0.5f, 0,0,-1, 0.5f, 0.5f,-0.5f, 0,0,-1,
     0.5f,-0.5f, 0.5f, 1,0,0,  0.5f,-0.5f,-0.5f, 1,0,0,  0.5f, 0.5f,-0.5f, 1,0,0,  0.5f, 0.5f, 0.5f, 1,0,0,
    -0.5f,-0.5f,-0.5f,-1,0,0, -0.5f,-0.5f, 0.5f,-1,0,0, -0.5f, 0.5f, 0.5f,-1,0,0, -0.5f, 0.5f,-0.5f,-1,0,0,
    -0.5f, 0.5f, 0.5f, 0,1,0,  0.5f, 0.5f, 0.5f, 0,1,0,  0.5f, 0.5f,-0.5f, 0,1,0, -0.5f, 0.5f,-0.5f, 0,1,0,
    -0.5f,-0.5f,-0.5f, 0,-1,0, 0.5f,-0.5f,-0.5f, 0,-1,0, 0.5f,-0.5f, 0.5f, 0,-1,0,-0.5f,-0.5f, 0.5f, 0,-1,0
};
static const unsigned short IDX[] = {
    0,1,2,0,2,3, 4,5,6,4,6,7, 8,9,10,8,10,11,
    12,13,14,12,14,15, 16,17,18,16,18,19, 20,21,22,20,22,23
};

static GLuint gProg,gVBO,gIBO;
static GLuint gSphereVBO=0, gSphereIBO=0;
static int gSphereIndexCount=0;
static GLint aPos,aNormal,uMVP,uModel,uColor;
static Mat4 gVP;
static double gLast=0, gT=0;

struct Palette { float torso[3]; float chest[3]; float helmet[3]; float belt[3]; float legs[3]; };
static Palette gPalettes[5] = {
    {{0.16f,0.42f,0.72f},{0.05f,0.75f,0.95f},{0.20f,0.28f,0.42f},{0.90f,0.55f,0.15f},{0.18f,0.22f,0.30f}},
    {{0.90f,0.32f,0.62f},{1.00f,0.20f,0.55f},{0.55f,0.14f,0.40f},{1.00f,0.80f,0.30f},{0.30f,0.12f,0.22f}},
    {{0.82f,0.88f,0.96f},{0.55f,0.82f,0.98f},{0.30f,0.38f,0.52f},{0.70f,0.76f,0.88f},{0.52f,0.58f,0.70f}},
    {{0.92f,0.40f,0.14f},{1.00f,0.75f,0.20f},{0.55f,0.20f,0.08f},{1.00f,0.90f,0.30f},{0.34f,0.14f,0.06f}},
    {{0.42f,0.22f,0.72f},{0.66f,0.38f,1.00f},{0.16f,0.08f,0.30f},{0.90f,0.60f,1.00f},{0.14f,0.08f,0.20f}}
};
static int gPaletteIdx = 0;

struct PetColors { float body[3]; float accent[3]; float eye[3]; };
static PetColors gPets[6] = {
    {{0,0,0},{0,0,0},{0,0,0}},
    {{0.35f,0.78f,1.00f},{0.10f,0.55f,0.90f},{0.05f,0.05f,0.09f}},
    {{0.62f,0.82f,1.00f},{0.35f,0.55f,0.85f},{0.05f,0.05f,0.09f}},
    {{1.00f,0.48f,0.24f},{0.90f,0.20f,0.10f},{0.05f,0.05f,0.09f}},
    {{1.00f,0.78f,0.34f},{0.90f,0.55f,0.15f},{0.05f,0.05f,0.09f}},
    {{0.50f,0.95f,1.00f},{0.10f,0.60f,0.85f},{0.05f,0.05f,0.09f}}
};
static int gPetIdx = 1;
static int gBagIdx = 0;
static int gGunIdx = -1;
static float gFOV = 60.0f;
static int   gAutoRotate = 0;
static float gDragSens = 1.0f;
static int gActiveSubject = 0;

extern "C" EMSCRIPTEN_KEEPALIVE
void set_character_preset(int idx){ if(idx<0)idx=0; if(idx>4)idx=4; gPaletteIdx=idx; }
extern "C" EMSCRIPTEN_KEEPALIVE
void set_pet_preset(int idx){ if(idx<0)idx=0; if(idx>5)idx=5; gPetIdx=idx; }
extern "C" EMSCRIPTEN_KEEPALIVE
void set_bag_preset(int idx){ if(idx<-1)idx=-1; if(idx>4)idx=4; gBagIdx=idx; }
extern "C" EMSCRIPTEN_KEEPALIVE
void set_gun_preset(int idx){ if(idx<-1)idx=-1; if(idx>4)idx=4; gGunIdx=idx; }
extern "C" EMSCRIPTEN_KEEPALIVE
void set_camera_fov(float deg){ if(deg<30)deg=30; if(deg>120)deg=120; gFOV=deg; }
extern "C" EMSCRIPTEN_KEEPALIVE
void set_drag_sensitivity(float s){ if(s<0.2f)s=0.2f; if(s>4.0f)s=4.0f; gDragSens=s; }
extern "C" EMSCRIPTEN_KEEPALIVE
void set_auto_rotate(int on){ gAutoRotate = on ? 1 : 0; }
extern "C" EMSCRIPTEN_KEEPALIVE
void set_active_subject(int s){ gActiveSubject = (s==1) ? 1 : 0; }

static float gCharYaw = 0.0f;
static float gPetYaw  = 0.0f;
static bool   gDragging = false;
static double gLastX = 0.0;

static EM_BOOL on_mouse_down(int, const EmscriptenMouseEvent* e, void*){
    if (e->button != 0) return EM_TRUE;
    gDragging = true; gLastX = e->clientX; return EM_TRUE;
}
static EM_BOOL on_mouse_up(int, const EmscriptenMouseEvent*, void*){ gDragging=false; return EM_TRUE; }
static EM_BOOL on_mouse_move(int, const EmscriptenMouseEvent* e, void*){
    if(!gDragging) return EM_TRUE;
    double dx = e->clientX - gLastX;
    gLastX = e->clientX;
    float delta = -(float)dx * 0.010f * gDragSens;
    if (gActiveSubject == 0) gCharYaw += delta;
    else                     gPetYaw  += delta;
    return EM_TRUE;
}
static EM_BOOL on_touch_start(int, const EmscriptenTouchEvent* e, void*){
    if (e->numTouches > 0){ gDragging = true; gLastX = e->touches[0].clientX; }
    return EM_TRUE;
}
static EM_BOOL on_touch_end(int, const EmscriptenTouchEvent*, void*){ gDragging=false; return EM_TRUE; }
static EM_BOOL on_touch_move(int, const EmscriptenTouchEvent* e, void*){
    if (!gDragging || e->numTouches < 1) return EM_TRUE;
    double dx = e->touches[0].clientX - gLastX;
    gLastX = e->touches[0].clientX;
    float delta = -(float)dx * 0.012f * gDragSens;
    if (gActiveSubject == 0) gCharYaw += delta;
    else                     gPetYaw  += delta;
    return EM_TRUE;
}

static GLuint sh(GLenum t,const char*src){
    GLuint s=glCreateShader(t);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){ char log[1024]; glGetShaderInfoLog(s,1024,nullptr,log); printf("Shader: %s\n",log); }
    return s;
}

static void setCubeAttribs(){
    glBindBuffer(GL_ARRAY_BUFFER, gVBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gIBO);
    glEnableVertexAttribArray(aPos);
    glVertexAttribPointer(aPos, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0);
    glEnableVertexAttribArray(aNormal);
    glVertexAttribPointer(aNormal, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float)));
}
static void setSphereAttribs(){
    glBindBuffer(GL_ARRAY_BUFFER, gSphereVBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gSphereIBO);
    glEnableVertexAttribArray(aPos);
    glVertexAttribPointer(aPos, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0);
    glEnableVertexAttribArray(aNormal);
    glVertexAttribPointer(aNormal, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float)));
}

static void drawBox(const Mat4&m,const float c[3]){
    Mat4 mvp=matMul(gVP,m);
    glUniformMatrix4fv(uModel,1,GL_FALSE,m.m);
    glUniformMatrix4fv(uMVP,1,GL_FALSE,mvp.m);
    glUniform3f(uColor,c[0],c[1],c[2]);
    setCubeAttribs();
    glDrawElements(GL_TRIANGLES,36,GL_UNSIGNED_SHORT,0);
}
static void drawSphere(const Mat4&m,const float c[3]){
    Mat4 mvp=matMul(gVP,m);
    glUniformMatrix4fv(uModel,1,GL_FALSE,m.m);
    glUniformMatrix4fv(uMVP,1,GL_FALSE,mvp.m);
    glUniform3f(uColor,c[0],c[1],c[2]);
    setSphereAttribs();
    glDrawElements(GL_TRIANGLES,gSphereIndexCount,GL_UNSIGNED_SHORT,0);
}
static void drawBoxRGB(const Mat4&m,float r,float g,float b){
    float c[3]={r,g,b};
    drawBox(m,c);
}

// Convenient shortcuts
static void XBOX(const Mat4& parent, float tx, float ty, float tz,
                 float sx, float sy, float sz, const float* c){
    drawBox(matMul(parent, matMul(matT(tx,ty,tz), matS(sx,sy,sz))), c);
}
static void XSPH(const Mat4& parent, float tx, float ty, float tz,
                 float sx, float sy, float sz, const float* c){
    drawSphere(matMul(parent, matMul(matT(tx,ty,tz), matS(sx,sy,sz))), c);
}

static void buildSphere(){
    const int RINGS = 12;
    const int SEGMENTS = 16;
    std::vector<float> verts;
    std::vector<unsigned short> idx;
    verts.reserve((RINGS+1)*(SEGMENTS+1)*6);
    idx.reserve(RINGS*SEGMENTS*6);
    for (int y=0; y<=RINGS; y++){
        float v = (float)y / RINGS;
        float phi = v * 3.14159265f;
        float sp = sinf(phi);
        float cp = cosf(phi);
        for (int x=0; x<=SEGMENTS; x++){
            float u = (float)x / SEGMENTS;
            float theta = u * 2.0f * 3.14159265f;
            float nx = sp * cosf(theta);
            float ny = cp;
            float nz = sp * sinf(theta);
            verts.push_back(nx*0.5f); verts.push_back(ny*0.5f); verts.push_back(nz*0.5f);
            verts.push_back(nx); verts.push_back(ny); verts.push_back(nz);
        }
    }
    for (int y=0; y<RINGS; y++){
        for (int x=0; x<SEGMENTS; x++){
            unsigned short a = (unsigned short)(y*(SEGMENTS+1)+x);
            unsigned short b = (unsigned short)(a+SEGMENTS+1);
            idx.push_back(a);   idx.push_back(b);   idx.push_back(a+1);
            idx.push_back(a+1); idx.push_back(b);   idx.push_back(b+1);
        }
    }
    gSphereIndexCount = (int)idx.size();
    glGenBuffers(1,&gSphereVBO);
    glBindBuffer(GL_ARRAY_BUFFER,gSphereVBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size()*sizeof(float), verts.data(), GL_STATIC_DRAW);
    glGenBuffers(1,&gSphereIBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,gSphereIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size()*sizeof(unsigned short), idx.data(), GL_STATIC_DRAW);
}

// ==== shared colors ====
static const float SKIN[3]      = {1.00f, 0.82f, 0.62f};
static const float SKIN_DK[3]   = {0.88f, 0.70f, 0.50f};
static const float EYE_W[3]     = {0.95f, 0.96f, 0.98f};
static const float EYE_D[3]     = {0.05f, 0.05f, 0.09f};
static const float BOOT_C[3]    = {0.05f, 0.06f, 0.08f};
static const float METAL_C[3]   = {0.16f, 0.20f, 0.26f};
static const float GLOW[3]      = {0.05f, 0.85f, 1.00f};

// ================================================================
//  CHARACTER 1 — ALPHA (blue soldier, basic)
// ================================================================
static void drawAlpha(const Mat4& root, Palette& P, float breathe, float sway){
    for (int side = -1; side <= 1; side += 2){
        float sx = side * 0.16f;
        XSPH(root, sx, 0.62f, 0.0f, 0.24f, 0.40f, 0.24f, P.legs);
        XSPH(root, sx, 0.38f, 0.02f, 0.22f, 0.14f, 0.22f, P.chest);
        XSPH(root, sx, 0.22f, 0.0f, 0.21f, 0.30f, 0.21f, P.legs);
        XBOX(root, sx, 0.08f, 0.02f, 0.22f, 0.14f, 0.26f, BOOT_C);
        XBOX(root, sx, 0.02f, 0.02f, 0.24f, 0.04f, 0.28f, METAL_C);
        XSPH(root, sx, 0.06f, 0.15f, 0.20f, 0.12f, 0.12f, BOOT_C);
    }
    XSPH(root, 0.0f, 0.92f, 0.0f, 0.62f, 0.24f, 0.36f, P.torso);
    XBOX(root, 0.0f, 0.94f, 0.0f, 0.62f, 0.08f, 0.36f, P.belt);
    XBOX(root, 0.0f, 0.94f, 0.20f, 0.14f, 0.10f, 0.06f, P.chest);
    XSPH(root, 0.0f, 1.14f, 0.0f, 0.58f, 0.28f, 0.34f, P.torso);
    XSPH(root, 0.0f, 1.40f + breathe*0.5f, 0.0f, 0.64f, 0.38f, 0.36f, P.torso);
    XSPH(root, 0.0f, 1.42f + breathe*0.5f, 0.16f, 0.48f, 0.30f, 0.12f, P.chest);
    XBOX(root, 0.0f, 1.42f + breathe*0.5f, 0.24f, 0.44f, 0.035f, 0.02f, GLOW);
    XSPH(root, -0.22f, 1.38f + breathe*0.5f, 0.0f, 0.10f, 0.52f, 0.36f, P.chest);
    XSPH(root,  0.22f, 1.38f + breathe*0.5f, 0.0f, 0.10f, 0.52f, 0.36f, P.chest);
    XSPH(root, 0.0f, 1.60f + breathe*0.5f, 0.0f, 0.30f, 0.12f, 0.32f, P.helmet);
    XSPH(root, 0.0f, 1.70f + breathe*0.5f, 0.0f, 0.20f, 0.16f, 0.20f, SKIN_DK);
    float hy = 1.92f + breathe;
    XSPH(root, 0.0f, hy, 0.0f, 0.46f, 0.48f, 0.46f, SKIN);
    XSPH(root, -0.22f, hy, 0.0f, 0.06f, 0.12f, 0.10f, SKIN);
    XSPH(root,  0.22f, hy, 0.0f, 0.06f, 0.12f, 0.10f, SKIN);
    XSPH(root, 0.0f, hy - 0.04f, 0.24f, 0.07f, 0.09f, 0.09f, SKIN_DK);
    XBOX(root, 0.0f, hy - 0.12f, 0.22f, 0.10f, 0.025f, 0.04f, EYE_D);
    XBOX(root, -0.10f, hy + 0.09f, 0.235f, 0.09f, 0.025f, 0.03f, EYE_D);
    XBOX(root,  0.10f, hy + 0.09f, 0.235f, 0.09f, 0.025f, 0.03f, EYE_D);
    XSPH(root, -0.10f, hy, 0.21f, 0.10f, 0.10f, 0.06f, EYE_W);
    XSPH(root,  0.10f, hy, 0.21f, 0.10f, 0.10f, 0.06f, EYE_W);
    XSPH(root, -0.10f, hy, 0.245f, 0.05f, 0.05f, 0.04f, EYE_D);
    XSPH(root,  0.10f, hy, 0.245f, 0.05f, 0.05f, 0.04f, EYE_D);
    XSPH(root, -0.24f, 1.96f + breathe, 0.0f, 0.08f, 0.24f, 0.44f, P.helmet);
    XSPH(root,  0.24f, 1.96f + breathe, 0.0f, 0.08f, 0.24f, 0.44f, P.helmet);
    XSPH(root, 0.0f, 1.96f + breathe, -0.20f, 0.48f, 0.28f, 0.14f, P.helmet);
    XBOX(root, 0.0f, 2.08f + breathe, 0.0f, 0.50f, 0.10f, 0.50f, P.helmet);
    XSPH(root, 0.0f, 2.16f + breathe, 0.0f, 0.52f, 0.22f, 0.52f, P.helmet);
    XBOX(root, 0.0f, 2.26f + breathe, -0.04f, 0.05f, 0.08f, 0.26f, P.chest);
    XBOX(root, 0.0f, 1.99f + breathe, 0.26f, 0.42f, 0.06f, 0.05f, GLOW);
    for (int side = -1; side <= 1; side += 2){
        float rot = side * (0.12f + sway);
        Mat4 ab = matMul(root, matT(side * 0.46f, 1.48f + breathe*0.5f, 0));
        Mat4 arm;
        if (gGunIdx >= 0){
            float fwd = (side > 0) ? -1.00f : -1.12f;
            float inw = -side * 0.28f;
            arm = matMul(ab, matRZ(inw));
            arm = matMul(arm, matRX(fwd));
        } else {
            arm = matMul(ab, matRZ(rot));
        }
        XSPH(ab, 0.0f, 0.02f, 0.0f, 0.34f, 0.26f, 0.36f, P.chest);
        XSPH(arm, 0.0f, -0.24f, 0.0f, 0.18f, 0.34f, 0.18f, P.torso);
        XSPH(arm, 0.0f, -0.44f, 0.02f, 0.16f, 0.12f, 0.18f, P.chest);
        XSPH(arm, 0.0f, -0.58f, 0.0f, 0.15f, 0.26f, 0.15f, SKIN);
        XBOX(arm, 0.0f, -0.72f, 0.0f, 0.16f, 0.04f, 0.16f, P.belt);
        XSPH(arm, 0.0f, -0.82f, 0.0f, 0.15f, 0.16f, 0.15f, SKIN);
        for (int f = 0; f < 4; f++){
            float fz = -0.055f + f * 0.037f;
            XBOX(arm, 0.0f, -0.92f, fz, 0.026f, 0.075f, 0.026f, SKIN);
            XSPH(arm, 0.0f, -0.97f, fz, 0.028f, 0.028f, 0.028f, SKIN);
        }
        XBOX(arm, -side * 0.09f, -0.82f, 0.02f, 0.045f, 0.055f, 0.045f, SKIN);
        XSPH(arm, -side * 0.11f, -0.87f, 0.02f, 0.045f, 0.045f, 0.045f, SKIN);
    }
}

// ================================================================
//  CHARACTER 2 — NOVA (female hero with long ponytail)
// ================================================================
static void drawNova(const Mat4& root, Palette& P, float breathe, float sway){
    float HAIR[3] = {0.55f, 0.12f, 0.32f};
    for (int side = -1; side <= 1; side += 2){
        float sx = side * 0.15f;
        XSPH(root, sx, 0.62f, 0.0f, 0.21f, 0.40f, 0.21f, P.legs);
        XSPH(root, sx, 0.38f, 0.02f, 0.19f, 0.14f, 0.19f, P.chest);
        XSPH(root, sx, 0.22f, 0.0f, 0.18f, 0.30f, 0.18f, P.legs);
        XBOX(root, sx, 0.08f, 0.02f, 0.20f, 0.14f, 0.24f, BOOT_C);
        XBOX(root, sx, 0.02f, 0.02f, 0.22f, 0.04f, 0.26f, METAL_C);
        XSPH(root, sx, 0.06f, 0.14f, 0.18f, 0.11f, 0.11f, BOOT_C);
    }
    XSPH(root, 0.0f, 0.92f, 0.0f, 0.58f, 0.26f, 0.34f, P.torso);
    XBOX(root, 0.0f, 0.94f, 0.0f, 0.56f, 0.08f, 0.34f, P.belt);
    XBOX(root, 0.0f, 0.94f, 0.18f, 0.12f, 0.10f, 0.06f, P.chest);
    XBOX(root, -0.28f, 0.88f, 0.0f, 0.08f, 0.12f, 0.14f, P.belt);
    XBOX(root,  0.28f, 0.88f, 0.0f, 0.08f, 0.12f, 0.14f, P.belt);
    XSPH(root, 0.0f, 1.14f, 0.0f, 0.52f, 0.28f, 0.32f, P.torso);
    XSPH(root, 0.0f, 1.40f + breathe*0.5f, 0.0f, 0.58f, 0.36f, 0.34f, P.torso);
    XSPH(root, 0.0f, 1.42f + breathe*0.5f, 0.16f, 0.44f, 0.28f, 0.12f, P.chest);
    XBOX(root, 0.0f, 1.42f + breathe*0.5f, 0.24f, 0.40f, 0.035f, 0.02f, GLOW);
    XSPH(root, -0.20f, 1.38f + breathe*0.5f, 0.0f, 0.08f, 0.50f, 0.34f, P.chest);
    XSPH(root,  0.20f, 1.38f + breathe*0.5f, 0.0f, 0.08f, 0.50f, 0.34f, P.chest);
    XSPH(root, 0.0f, 1.60f + breathe*0.5f, 0.0f, 0.26f, 0.10f, 0.28f, P.helmet);
    XSPH(root, 0.0f, 1.70f + breathe*0.5f, 0.0f, 0.16f, 0.14f, 0.16f, SKIN_DK);
    float hy = 1.92f + breathe;
    XSPH(root, 0.0f, hy, 0.0f, 0.44f, 0.46f, 0.44f, SKIN);
    XSPH(root, -0.21f, hy, 0.0f, 0.05f, 0.11f, 0.09f, SKIN);
    XSPH(root,  0.21f, hy, 0.0f, 0.05f, 0.11f, 0.09f, SKIN);
    XSPH(root, 0.0f, hy - 0.04f, 0.23f, 0.06f, 0.08f, 0.08f, SKIN_DK);
    XBOX(root, 0.0f, hy - 0.12f, 0.21f, 0.09f, 0.025f, 0.04f, EYE_D);
    XBOX(root, -0.10f, hy + 0.09f, 0.225f, 0.09f, 0.025f, 0.03f, EYE_D);
    XBOX(root,  0.10f, hy + 0.09f, 0.225f, 0.09f, 0.025f, 0.03f, EYE_D);
    XSPH(root, -0.10f, hy, 0.20f, 0.09f, 0.09f, 0.06f, EYE_W);
    XSPH(root,  0.10f, hy, 0.20f, 0.09f, 0.09f, 0.06f, EYE_W);
    XSPH(root, -0.10f, hy, 0.235f, 0.045f, 0.045f, 0.04f, EYE_D);
    XSPH(root,  0.10f, hy, 0.235f, 0.045f, 0.045f, 0.04f, EYE_D);
    // LONG HAIR — thick behind head + ponytail going down back
    XSPH(root, 0.0f, hy + 0.04f, -0.20f, 0.46f, 0.46f, 0.22f, HAIR);
    XSPH(root, 0.0f, hy - 0.24f, -0.24f, 0.24f, 0.36f, 0.18f, HAIR);
    XSPH(root, 0.0f, hy - 0.60f, -0.22f, 0.18f, 0.32f, 0.14f, HAIR);
    XSPH(root, 0.0f, hy - 0.94f, -0.20f, 0.12f, 0.24f, 0.10f, HAIR);
    // slim crown helmet with V-visor
    XSPH(root, -0.22f, 2.02f + breathe, -0.04f, 0.06f, 0.18f, 0.38f, P.helmet);
    XSPH(root,  0.22f, 2.02f + breathe, -0.04f, 0.06f, 0.18f, 0.38f, P.helmet);
    XSPH(root, 0.0f, 2.14f + breathe, 0.0f, 0.48f, 0.16f, 0.46f, P.helmet);
    XSPH(root, 0.0f, 2.22f + breathe, 0.0f, 0.30f, 0.14f, 0.32f, P.helmet);
    // side crown spikes
    XSPH(root, -0.28f, 2.20f + breathe, 0.0f, 0.05f, 0.14f, 0.05f, P.chest);
    XSPH(root,  0.28f, 2.20f + breathe, 0.0f, 0.05f, 0.14f, 0.05f, P.chest);
    // V visor
    XBOX(root, -0.12f, 1.98f + breathe, 0.24f, 0.30f, 0.06f, 0.04f, GLOW);
    XBOX(root,  0.12f, 1.98f + breathe, 0.24f, 0.30f, 0.06f, 0.04f, GLOW);
    for (int side = -1; side <= 1; side += 2){
        float rot = side * (0.12f + sway);
        Mat4 ab = matMul(root, matT(side * 0.44f, 1.48f + breathe*0.5f, 0));
        Mat4 arm;
        if (gGunIdx >= 0){
            float fwd = (side > 0) ? -1.00f : -1.12f;
            float inw = -side * 0.28f;
            arm = matMul(ab, matRZ(inw));
            arm = matMul(arm, matRX(fwd));
        } else {
            arm = matMul(ab, matRZ(rot));
        }
        XSPH(ab, 0.0f, 0.02f, 0.0f, 0.28f, 0.22f, 0.30f, P.chest);
        XSPH(arm, 0.0f, -0.24f, 0.0f, 0.15f, 0.34f, 0.15f, P.torso);
        XSPH(arm, 0.0f, -0.44f, 0.02f, 0.13f, 0.11f, 0.15f, P.chest);
        XSPH(arm, 0.0f, -0.58f, 0.0f, 0.12f, 0.26f, 0.12f, SKIN);
        XBOX(arm, 0.0f, -0.72f, 0.0f, 0.13f, 0.04f, 0.13f, P.belt);
        XSPH(arm, 0.0f, -0.82f, 0.0f, 0.13f, 0.14f, 0.13f, SKIN);
        for (int f = 0; f < 4; f++){
            float fz = -0.05f + f * 0.034f;
            XBOX(arm, 0.0f, -0.90f, fz, 0.022f, 0.065f, 0.022f, SKIN);
            XSPH(arm, 0.0f, -0.94f, fz, 0.024f, 0.024f, 0.024f, SKIN);
        }
        XBOX(arm, -side * 0.08f, -0.82f, 0.02f, 0.04f, 0.05f, 0.04f, SKIN);
        XSPH(arm, -side * 0.10f, -0.86f, 0.02f, 0.04f, 0.04f, 0.04f, SKIN);
    }
}

// ================================================================
//  CHARACTER 3 — GHOST (hooded with cape, glowing eyes)
// ================================================================
static void drawGhost(const Mat4& root, Palette& P, float breathe, float sway){
    for (int side = -1; side <= 1; side += 2){
        float sx = side * 0.15f;
        XSPH(root, sx, 0.62f, 0.0f, 0.22f, 0.40f, 0.22f, P.torso);
        XSPH(root, sx, 0.38f, 0.02f, 0.20f, 0.14f, 0.20f, P.helmet);
        XSPH(root, sx, 0.22f, 0.0f, 0.19f, 0.30f, 0.19f, P.torso);
        XBOX(root, sx, 0.08f, 0.02f, 0.22f, 0.14f, 0.26f, BOOT_C);
        XBOX(root, sx, 0.02f, 0.02f, 0.24f, 0.04f, 0.28f, METAL_C);
    }
    XSPH(root, 0.0f, 0.92f, 0.0f, 0.58f, 0.26f, 0.34f, P.torso);
    XBOX(root, 0.0f, 0.94f, 0.0f, 0.58f, 0.06f, 0.34f, P.belt);
    XSPH(root, 0.0f, 1.20f, 0.0f, 0.56f, 0.36f, 0.34f, P.torso);
    XSPH(root, 0.0f, 1.50f + breathe*0.5f, 0.0f, 0.58f, 0.36f, 0.34f, P.torso);
    XSPH(root, 0.0f, 1.55f + breathe*0.5f, 0.16f, 0.42f, 0.24f, 0.10f, P.chest);
    XBOX(root, 0.0f, 1.50f + breathe*0.5f, 0.22f, 0.38f, 0.03f, 0.02f, GLOW);

    // CAPE — long flowing behind
    Mat4 capeBase = matMul(root, matT(0.0f, 1.50f, -0.16f));
    float cs = sway * 3.0f;
    XSPH(capeBase, 0.0f,  0.00f, 0.0f, 0.62f, 0.50f, 0.10f, P.helmet);
    XSPH(capeBase, cs*0.05f, -0.42f, 0.0f, 0.58f, 0.44f, 0.09f, P.helmet);
    XSPH(capeBase, cs*0.10f, -0.82f, 0.0f, 0.54f, 0.40f, 0.08f, P.helmet);
    XSPH(capeBase, cs*0.15f, -1.16f, 0.0f, 0.48f, 0.32f, 0.06f, P.helmet);

    // HOOD — big rounded shape covering face
    float hy = 1.94f + breathe;
    XSPH(root, 0.0f, hy, 0.0f, 0.40f, 0.42f, 0.40f, P.torso);
    XSPH(root, 0.0f, hy + 0.02f, 0.02f, 0.50f, 0.52f, 0.50f, P.helmet);
    XSPH(root, 0.0f, hy + 0.10f, 0.16f, 0.52f, 0.24f, 0.28f, P.helmet);
    XSPH(root, 0.0f, hy + 0.12f, -0.18f, 0.42f, 0.28f, 0.20f, P.helmet);
    // glowing eye slit
    XSPH(root, -0.11f, hy - 0.02f, 0.22f, 0.05f, 0.05f, 0.04f, GLOW);
    XSPH(root,  0.11f, hy - 0.02f, 0.22f, 0.05f, 0.05f, 0.04f, GLOW);

    for (int side = -1; side <= 1; side += 2){
        float rot = side * (0.10f + sway);
        Mat4 ab = matMul(root, matT(side * 0.42f, 1.55f + breathe*0.5f, 0));
        Mat4 arm;
        if (gGunIdx >= 0){
            float fwd = (side > 0) ? -1.00f : -1.12f;
            float inw = -side * 0.28f;
            arm = matMul(ab, matRZ(inw));
            arm = matMul(arm, matRX(fwd));
        } else {
            arm = matMul(ab, matRZ(rot));
        }
        XSPH(ab, 0.0f, 0.0f, 0.0f, 0.30f, 0.24f, 0.32f, P.chest);
        XSPH(arm, 0.0f, -0.24f, 0.0f, 0.14f, 0.32f, 0.14f, P.torso);
        XSPH(arm, 0.0f, -0.44f, 0.0f, 0.13f, 0.11f, 0.14f, P.helmet);
        XSPH(arm, 0.0f, -0.60f, 0.0f, 0.13f, 0.26f, 0.13f, P.torso);
        XSPH(arm, 0.0f, -0.80f, 0.0f, 0.14f, 0.14f, 0.14f, P.helmet);
        for (int f = 0; f < 4; f++){
            float fz = -0.05f + f * 0.034f;
            XBOX(arm, 0.0f, -0.90f, fz, 0.022f, 0.06f, 0.022f, P.helmet);
        }
    }
}

// ================================================================
//  CHARACTER 4 — BLAZE (bulky fire warrior with horns)
// ================================================================
static void drawBlaze(const Mat4& root, Palette& P, float breathe, float sway){
    // thick legs
    for (int side = -1; side <= 1; side += 2){
        float sx = side * 0.18f;
        XSPH(root, sx, 0.60f, 0.0f, 0.28f, 0.42f, 0.28f, P.legs);
        XSPH(root, sx, 0.36f, 0.02f, 0.26f, 0.16f, 0.26f, P.chest);
        XSPH(root, sx, 0.20f, 0.0f, 0.26f, 0.32f, 0.26f, P.legs);
        XBOX(root, sx, 0.08f, 0.03f, 0.30f, 0.18f, 0.34f, BOOT_C);
        XBOX(root, sx, 0.00f, 0.03f, 0.32f, 0.04f, 0.36f, METAL_C);
        XSPH(root, sx, 0.08f, 0.20f, 0.24f, 0.16f, 0.14f, BOOT_C);
    }
    // bulk torso
    XSPH(root, 0.0f, 0.94f, 0.0f, 0.70f, 0.28f, 0.42f, P.torso);
    XBOX(root, 0.0f, 0.92f, 0.0f, 0.72f, 0.10f, 0.44f, P.belt);
    XBOX(root, 0.0f, 0.92f, 0.24f, 0.18f, 0.12f, 0.06f, P.chest);
    XSPH(root, 0.0f, 1.18f, 0.0f, 0.68f, 0.34f, 0.42f, P.torso);
    XSPH(root, 0.0f, 1.48f + breathe*0.5f, 0.0f, 0.76f, 0.44f, 0.46f, P.torso);
    XSPH(root, 0.0f, 1.52f + breathe*0.5f, 0.20f, 0.60f, 0.36f, 0.14f, P.chest);
    XSPH(root, 0.0f, 1.62f + breathe*0.5f, 0.18f, 0.44f, 0.18f, 0.10f, P.belt);
    XBOX(root, 0.0f, 1.50f + breathe*0.5f, 0.30f, 0.52f, 0.04f, 0.02f, GLOW);
    XSPH(root, -0.28f, 1.48f + breathe*0.5f, 0.0f, 0.14f, 0.58f, 0.46f, P.chest);
    XSPH(root,  0.28f, 1.48f + breathe*0.5f, 0.0f, 0.14f, 0.58f, 0.46f, P.chest);
    XSPH(root, 0.0f, 1.72f + breathe*0.5f, 0.0f, 0.36f, 0.14f, 0.38f, P.helmet);
    XSPH(root, 0.0f, 1.80f + breathe*0.5f, 0.0f, 0.24f, 0.14f, 0.24f, SKIN_DK);
    float hy = 2.00f + breathe;
    XSPH(root, 0.0f, hy, 0.0f, 0.48f, 0.48f, 0.48f, SKIN);
    XSPH(root, -0.23f, hy, 0.0f, 0.08f, 0.14f, 0.12f, SKIN);
    XSPH(root,  0.23f, hy, 0.0f, 0.08f, 0.14f, 0.12f, SKIN);
    XSPH(root, 0.0f, hy - 0.04f, 0.26f, 0.08f, 0.10f, 0.10f, SKIN_DK);
    XBOX(root, 0.0f, hy - 0.13f, 0.24f, 0.12f, 0.03f, 0.05f, EYE_D);
    XBOX(root, -0.11f, hy + 0.08f, 0.255f, 0.10f, 0.03f, 0.03f, EYE_D);
    XBOX(root,  0.11f, hy + 0.08f, 0.255f, 0.10f, 0.03f, 0.03f, EYE_D);
    float fireEye[3] = {1.00f, 0.45f, 0.10f};
    XSPH(root, -0.11f, hy, 0.23f, 0.09f, 0.09f, 0.05f, fireEye);
    XSPH(root,  0.11f, hy, 0.23f, 0.09f, 0.09f, 0.05f, fireEye);
    XSPH(root, -0.11f, hy, 0.255f, 0.04f, 0.04f, 0.03f, EYE_D);
    XSPH(root,  0.11f, hy, 0.255f, 0.04f, 0.04f, 0.03f, EYE_D);
    // horned helmet
    XBOX(root, 0.0f, hy + 0.16f, 0.0f, 0.54f, 0.14f, 0.54f, P.helmet);
    XSPH(root, 0.0f, hy + 0.26f, 0.0f, 0.52f, 0.20f, 0.52f, P.helmet);
    XBOX(root, 0.0f, hy + 0.38f, -0.02f, 0.08f, 0.16f, 0.36f, P.chest);
    // big front crest
    XSPH(root, -0.26f, hy + 0.28f, -0.04f, 0.10f, 0.14f, 0.14f, P.helmet);
    XSPH(root, -0.36f, hy + 0.34f, -0.06f, 0.08f, 0.16f, 0.12f, P.helmet);
    XSPH(root, -0.44f, hy + 0.42f, -0.08f, 0.05f, 0.12f, 0.06f, P.chest);
    XSPH(root,  0.26f, hy + 0.28f, -0.04f, 0.10f, 0.14f, 0.14f, P.helmet);
    XSPH(root,  0.36f, hy + 0.34f, -0.06f, 0.08f, 0.16f, 0.12f, P.helmet);
    XSPH(root,  0.44f, hy + 0.42f, -0.08f, 0.05f, 0.12f, 0.06f, P.chest);
    XBOX(root, 0.0f, hy + 0.02f, 0.27f, 0.42f, 0.06f, 0.05f, GLOW);
    // thick arms
    for (int side = -1; side <= 1; side += 2){
        float rot = side * (0.10f + sway);
        Mat4 ab = matMul(root, matT(side * 0.54f, 1.56f + breathe*0.5f, 0));
        Mat4 arm;
        if (gGunIdx >= 0){
            float fwd = (side > 0) ? -1.00f : -1.12f;
            float inw = -side * 0.28f;
            arm = matMul(ab, matRZ(inw));
            arm = matMul(arm, matRX(fwd));
        } else {
            arm = matMul(ab, matRZ(rot));
        }
        XSPH(ab, 0.0f, 0.02f, 0.0f, 0.42f, 0.32f, 0.44f, P.chest);
        // SHOULDER SPIKE
        XSPH(ab, side * 0.14f, 0.24f, -0.06f, 0.12f, 0.20f, 0.12f, P.belt);
        XSPH(ab, side * 0.22f, 0.38f, -0.10f, 0.08f, 0.18f, 0.08f, P.chest);
        XSPH(ab, side * 0.28f, 0.50f, -0.14f, 0.05f, 0.14f, 0.05f, P.belt);
        XSPH(arm, 0.0f, -0.26f, 0.0f, 0.22f, 0.36f, 0.22f, P.torso);
        XSPH(arm, 0.0f, -0.48f, 0.02f, 0.20f, 0.14f, 0.22f, P.chest);
        XSPH(arm, 0.0f, -0.64f, 0.0f, 0.20f, 0.28f, 0.20f, P.torso);
        XSPH(arm, side * 0.14f, -0.62f, 0.0f, 0.06f, 0.14f, 0.06f, P.belt);
        XBOX(arm, 0.0f, -0.82f, 0.0f, 0.20f, 0.05f, 0.20f, P.belt);
        XSPH(arm, 0.0f, -0.92f, 0.0f, 0.19f, 0.18f, 0.19f, SKIN);
        for (int f = 0; f < 4; f++){
            float fz = -0.07f + f * 0.047f;
            XBOX(arm, 0.0f, -1.04f, fz, 0.032f, 0.075f, 0.032f, SKIN);
            XSPH(arm, 0.0f, -1.09f, fz, 0.032f, 0.032f, 0.032f, SKIN);
        }
        XBOX(arm, -side * 0.12f, -0.92f, 0.02f, 0.055f, 0.065f, 0.055f, SKIN);
        XSPH(arm, -side * 0.14f, -0.98f, 0.02f, 0.055f, 0.055f, 0.055f, SKIN);
    }
}

// ================================================================
//  CHARACTER 5 — SHADOW (ninja with hood, mask, sash, back blades)
// ================================================================
static void drawShadow(const Mat4& root, Palette& P, float breathe, float sway){
    for (int side = -1; side <= 1; side += 2){
        float sx = side * 0.15f;
        XSPH(root, sx, 0.62f, 0.0f, 0.20f, 0.40f, 0.20f, P.legs);
        XBOX(root, sx, 0.52f, 0.0f, 0.22f, 0.04f, 0.22f, P.belt);
        XSPH(root, sx, 0.38f, 0.02f, 0.18f, 0.13f, 0.18f, P.chest);
        XBOX(root, sx, 0.24f, 0.0f, 0.21f, 0.04f, 0.21f, P.belt);
        XSPH(root, sx, 0.22f, 0.0f, 0.18f, 0.30f, 0.18f, P.legs);
        XBOX(root, sx, 0.08f, 0.02f, 0.20f, 0.14f, 0.26f, BOOT_C);
        XBOX(root, sx, 0.02f, 0.02f, 0.22f, 0.04f, 0.28f, METAL_C);
        XSPH(root, sx, 0.06f, 0.14f, 0.18f, 0.11f, 0.11f, BOOT_C);
    }
    XSPH(root, 0.0f, 0.92f, 0.0f, 0.56f, 0.22f, 0.32f, P.torso);
    XBOX(root, 0.0f, 0.94f, 0.0f, 0.54f, 0.06f, 0.32f, P.belt);
    XSPH(root, 0.0f, 1.14f, 0.0f, 0.50f, 0.28f, 0.30f, P.torso);
    XSPH(root, 0.0f, 1.42f + breathe*0.5f, 0.0f, 0.58f, 0.38f, 0.34f, P.torso);
    XSPH(root, 0.0f, 1.44f + breathe*0.5f, 0.16f, 0.42f, 0.26f, 0.10f, P.chest);
    XBOX(root, 0.0f, 1.44f + breathe*0.5f, 0.22f, 0.36f, 0.03f, 0.02f, GLOW);
    // SASH — diagonal
    Mat4 sash = matMul(root, matT(0.0f, 1.30f, 0.16f));
    sash = matMul(sash, matRZ(-0.5f));
    XBOX(sash, 0.0f, 0.0f, 0.0f, 0.10f, 0.62f, 0.06f, P.belt);
    XSPH(root, 0.0f, 1.60f + breathe*0.5f, 0.0f, 0.26f, 0.12f, 0.28f, P.helmet);
    float hy = 1.94f + breathe;
    XSPH(root, 0.0f, hy, 0.0f, 0.42f, 0.44f, 0.42f, P.torso);
    // hood pointed back
    XSPH(root, 0.0f, hy + 0.06f, -0.02f, 0.48f, 0.50f, 0.46f, P.helmet);
    XSPH(root, 0.0f, hy + 0.14f, -0.22f, 0.34f, 0.30f, 0.22f, P.helmet);
    XSPH(root, 0.0f, hy + 0.10f, -0.34f, 0.18f, 0.16f, 0.14f, P.helmet);
    // FACE MASK
    XSPH(root, 0.0f, hy - 0.08f, 0.24f, 0.36f, 0.20f, 0.10f, P.belt);
    XSPH(root, 0.0f, hy - 0.10f, 0.26f, 0.28f, 0.14f, 0.08f, P.helmet);
    // glowing eyes only
    XSPH(root, -0.10f, hy + 0.02f, 0.235f, 0.06f, 0.03f, 0.03f, GLOW);
    XSPH(root,  0.10f, hy + 0.02f, 0.235f, 0.06f, 0.03f, 0.03f, GLOW);
    for (int side = -1; side <= 1; side += 2){
        float rot = side * (0.10f + sway);
        Mat4 ab = matMul(root, matT(side * 0.42f, 1.50f + breathe*0.5f, 0));
        Mat4 arm;
        if (gGunIdx >= 0){
            float fwd = (side > 0) ? -1.00f : -1.12f;
            float inw = -side * 0.28f;
            arm = matMul(ab, matRZ(inw));
            arm = matMul(arm, matRX(fwd));
        } else {
            arm = matMul(ab, matRZ(rot));
        }
        XSPH(ab, 0.0f, 0.02f, 0.0f, 0.28f, 0.22f, 0.32f, P.chest);
        // BACK BLADE
        XSPH(ab, 0.0f, 0.14f, -0.14f, 0.06f, 0.18f, 0.05f, P.belt);
        XSPH(ab, 0.0f, 0.28f, -0.24f, 0.04f, 0.14f, 0.04f, P.chest);
        XSPH(arm, 0.0f, -0.24f, 0.0f, 0.15f, 0.32f, 0.15f, P.torso);
        XSPH(arm, 0.0f, -0.44f, 0.0f, 0.13f, 0.11f, 0.15f, P.chest);
        XSPH(arm, 0.0f, -0.60f, 0.0f, 0.13f, 0.26f, 0.13f, P.torso);
        XBOX(arm, 0.0f, -0.72f, 0.0f, 0.14f, 0.04f, 0.14f, P.belt);
        XSPH(arm, 0.0f, -0.82f, 0.0f, 0.13f, 0.14f, 0.13f, P.torso);
        for (int f = 0; f < 4; f++){
            float fz = -0.05f + f * 0.034f;
            XBOX(arm, 0.0f, -0.90f, fz, 0.022f, 0.065f, 0.022f, P.torso);
        }
        XBOX(arm, -side * 0.08f, -0.82f, 0.02f, 0.04f, 0.05f, 0.04f, P.torso);
    }
}

// ================================================================
//  PET
// ================================================================

// ================================================================
//  BACKPACK — attached to character's back
// ================================================================
struct BagColors { float main[3]; float accent[3]; float strap[3]; };
static BagColors gBags[5] = {
    {{0.48f,0.63f,0.78f},{0.30f,0.42f,0.56f},{0.18f,0.22f,0.28f}},
    {{0.25f,0.66f,1.00f},{0.10f,0.42f,0.75f},{0.10f,0.15f,0.22f}},
    {{0.66f,0.44f,1.00f},{0.42f,0.22f,0.80f},{0.16f,0.10f,0.28f}},
    {{1.00f,0.72f,0.30f},{0.85f,0.52f,0.12f},{0.35f,0.22f,0.06f}},
    {{1.00f,0.35f,0.62f},{0.80f,0.15f,0.42f},{0.30f,0.05f,0.16f}}
};


static void drawBackpack(const Mat4& root){
    if (gBagIdx < 0) return;
    BagColors& B = gBags[gBagIdx];
    Mat4 R = matMul(root, matT(0.0f, 1.38f, -0.30f));

    if (gBagIdx == 0){
        // ============ SCOUT PACK (compact tactical) ============
        XBOX(R, 0.0f,  0.00f, -0.12f, 0.44f, 0.50f, 0.20f, B.main);
        // chamfer strips (cut edges)
        XBOX(R, 0.0f,  0.24f, -0.12f, 0.36f, 0.04f, 0.20f, B.accent);
        XBOX(R, 0.0f, -0.24f, -0.12f, 0.36f, 0.04f, 0.20f, B.accent);
        XBOX(R, -0.20f, 0.00f, -0.12f, 0.04f, 0.42f, 0.20f, B.accent);
        XBOX(R,  0.20f, 0.00f, -0.12f, 0.04f, 0.42f, 0.20f, B.accent);
        // angled front flap
        Mat4 flap = matMul(R, matT(0.0f, 0.06f, -0.23f));
        flap = matMul(flap, matRX(0.20f));
        XBOX(flap, 0.0f, 0.0f, 0.0f, 0.38f, 0.20f, 0.04f, B.accent);
        // angled top lid
        Mat4 lid = matMul(R, matT(0.0f, 0.27f, -0.10f));
        lid = matMul(lid, matRX(-0.25f));
        XBOX(lid, 0.0f, 0.0f, 0.0f, 0.42f, 0.05f, 0.22f, B.accent);
        // buckle
        XBOX(R, 0.0f, -0.06f, -0.26f, 0.06f, 0.07f, 0.03f, B.strap);
        XBOX(R, 0.0f, -0.02f, -0.26f, 0.10f, 0.025f, 0.03f, B.strap);
        // top carry handle
        XBOX(R, 0.0f, 0.34f, -0.08f, 0.14f, 0.05f, 0.06f, B.strap);
        XSPH(R, -0.06f, 0.32f, -0.08f, 0.05f, 0.06f, 0.06f, B.accent);
        XSPH(R,  0.06f, 0.32f, -0.08f, 0.05f, 0.06f, 0.06f, B.accent);
        // side pockets
        XBOX(R, -0.25f, -0.04f, -0.12f, 0.05f, 0.22f, 0.16f, B.accent);
        XBOX(R,  0.25f, -0.04f, -0.12f, 0.05f, 0.22f, 0.16f, B.accent);
        XBOX(R, -0.26f, -0.12f, -0.12f, 0.02f, 0.04f, 0.16f, B.strap);
        XBOX(R,  0.26f, -0.12f, -0.12f, 0.02f, 0.04f, 0.16f, B.strap);
        // molle straps (3 horizontal thin)
        for (int i = 0; i < 3; i++){
            XBOX(R, 0.0f, -0.14f + i*0.11f, -0.23f, 0.34f, 0.014f, 0.012f, B.strap);
        }
        // shoulder straps
        XBOX(R, -0.21f, 0.16f, 0.10f, 0.06f, 0.56f, 0.05f, B.strap);
        XBOX(R,  0.21f, 0.16f, 0.10f, 0.06f, 0.56f, 0.05f, B.strap);
    }

    else if (gBagIdx == 1){
        // ============ TACTICAL BAG (bigger, layered) ============
        XBOX(R, 0.0f, -0.14f, -0.12f, 0.52f, 0.22f, 0.22f, B.main);  // bottom (wide)
        XBOX(R, 0.0f,  0.06f, -0.12f, 0.48f, 0.22f, 0.22f, B.main);  // mid
        XBOX(R, 0.0f,  0.24f, -0.12f, 0.42f, 0.18f, 0.22f, B.main);  // top (narrow)
        // bevel rings between layers
        XBOX(R, 0.0f, -0.02f, -0.12f, 0.50f, 0.02f, 0.24f, B.accent);
        XBOX(R, 0.0f,  0.16f, -0.12f, 0.46f, 0.02f, 0.24f, B.accent);
        // angled top lid
        Mat4 lid = matMul(R, matT(0.0f, 0.34f, -0.12f));
        lid = matMul(lid, matRX(-0.35f));
        XBOX(lid, 0.0f, 0.0f, 0.0f, 0.40f, 0.06f, 0.22f, B.accent);
        // front grid panel
        XBOX(R, 0.0f, -0.06f, -0.25f, 0.34f, 0.36f, 0.04f, B.accent);
        for (int i = 0; i < 4; i++)
            XBOX(R, 0.0f, -0.20f + i*0.09f, -0.27f, 0.30f, 0.008f, 0.008f, B.strap);
        for (int i = 0; i < 3; i++)
            XBOX(R, -0.12f + i*0.12f, -0.06f, -0.27f, 0.008f, 0.34f, 0.008f, B.strap);
        // central big buckle
        XBOX(R, 0.0f, 0.0f, -0.27f, 0.10f, 0.10f, 0.03f, B.strap);
        XSPH(R, 0.0f, 0.0f, -0.29f, 0.06f, 0.06f, 0.02f, B.main);
        // top handle (thick)
        XBOX(R, 0.0f, 0.40f, -0.08f, 0.18f, 0.06f, 0.08f, B.strap);
        XSPH(R, -0.09f, 0.38f, -0.08f, 0.05f, 0.06f, 0.06f, B.accent);
        XSPH(R,  0.09f, 0.38f, -0.08f, 0.05f, 0.06f, 0.06f, B.accent);
        // side mesh pouches
        for (int sd = -1; sd <= 1; sd += 2){
            XBOX(R, sd * 0.29f, -0.06f, -0.12f, 0.06f, 0.30f, 0.20f, B.accent);
            for (int i = 0; i < 4; i++)
                XBOX(R, sd * 0.32f, -0.16f + i*0.08f, -0.12f, 0.01f, 0.05f, 0.16f, B.strap);
        }
        // X cross straps on back
        Mat4 xs = matMul(R, matT(0.0f, 0.0f, -0.03f));
        XBOX(matMul(xs, matRZ( 0.7f)), 0.0f, 0.0f, 0.0f, 0.10f, 0.55f, 0.03f, B.strap);
        XBOX(matMul(xs, matRZ(-0.7f)), 0.0f, 0.0f, 0.0f, 0.10f, 0.55f, 0.03f, B.strap);
        // shoulder harness
        XBOX(R, -0.24f, 0.18f, 0.12f, 0.07f, 0.60f, 0.06f, B.strap);
        XBOX(R,  0.24f, 0.18f, 0.12f, 0.07f, 0.60f, 0.06f, B.strap);
    }

    else if (gBagIdx == 2){
        // ============ CYBERPACK (angular, futuristic) ============
        float cy[3] = {0.05f, 0.90f, 1.00f};
        XBOX(R, 0.0f, 0.0f, -0.12f, 0.44f, 0.56f, 0.22f, B.main);
        // angled top cut
        Mat4 topc = matMul(R, matT(0.0f, 0.28f, -0.12f));
        topc = matMul(topc, matRX(-0.5f));
        XBOX(topc, 0.0f, 0.0f, 0.0f, 0.42f, 0.08f, 0.22f, B.accent);
        // angled bottom cut
        Mat4 botc = matMul(R, matT(0.0f, -0.28f, -0.12f));
        botc = matMul(botc, matRX(0.5f));
        XBOX(botc, 0.0f, 0.0f, 0.0f, 0.42f, 0.08f, 0.22f, B.accent);
        // carbon fiber stripes (front)
        for (int i = 0; i < 5; i++)
            XBOX(R, 0.0f, -0.20f + i*0.10f, -0.23f, 0.36f, 0.008f, 0.008f, B.strap);
        // glowing vertical spine
        XBOX(R, 0.0f, 0.0f, -0.24f, 0.05f, 0.48f, 0.02f, cy);
        XSPH(R, 0.0f, -0.20f, -0.25f, 0.04f, 0.04f, 0.03f, cy);
        XSPH(R, 0.0f,  0.20f, -0.25f, 0.04f, 0.04f, 0.03f, cy);
        // side angular panels
        XBOX(R, -0.24f, 0.0f, -0.12f, 0.06f, 0.34f, 0.20f, B.accent);
        XBOX(R,  0.24f, 0.0f, -0.12f, 0.06f, 0.34f, 0.20f, B.accent);
        // LED indicators (left side)
        for (int i = 0; i < 3; i++)
            XBOX(R, -0.27f, -0.12f + i*0.10f, -0.12f, 0.01f, 0.03f, 0.05f, cy);
        // antenna
        XBOX(R, 0.16f, 0.34f, -0.06f, 0.025f, 0.20f, 0.025f, B.accent);
        XSPH(R, 0.16f, 0.46f, -0.06f, 0.04f, 0.04f, 0.04f, cy);
        // top handle
        XBOX(R, 0.0f, 0.36f, -0.10f, 0.16f, 0.04f, 0.06f, B.strap);
        // harness
        XBOX(R, -0.22f, 0.16f, 0.10f, 0.06f, 0.58f, 0.05f, B.strap);
        XBOX(R,  0.22f, 0.16f, 0.10f, 0.06f, 0.58f, 0.05f, B.strap);
    }

    else if (gBagIdx == 3){
        // ============ ELITE CARRIER (large military) ============
        XBOX(R, 0.0f, -0.16f, -0.14f, 0.56f, 0.26f, 0.24f, B.main);  // base
        XBOX(R, 0.0f,  0.08f, -0.14f, 0.52f, 0.24f, 0.24f, B.main);  // mid
        XBOX(R, 0.0f,  0.28f, -0.14f, 0.44f, 0.20f, 0.22f, B.main);  // top
        // bevel rings
        XBOX(R, 0.0f, -0.03f, -0.14f, 0.54f, 0.02f, 0.26f, B.accent);
        XBOX(R, 0.0f,  0.20f, -0.14f, 0.48f, 0.02f, 0.25f, B.accent);
        // rolled top (rounded cylinder)
        XSPH(R, 0.0f, 0.40f, -0.12f, 0.48f, 0.12f, 0.22f, B.accent);
        XBOX(R, 0.0f, 0.44f, -0.12f, 0.36f, 0.06f, 0.20f, B.main);
        // big central buckle
        XBOX(R, 0.0f, 0.08f, -0.27f, 0.12f, 0.12f, 0.04f, B.strap);
        XSPH(R, 0.0f, 0.08f, -0.29f, 0.07f, 0.07f, 0.025f, B.main);
        XBOX(R, 0.0f, -0.16f, -0.27f, 0.16f, 0.05f, 0.03f, B.strap);
        XBOX(R, 0.0f,  0.28f, -0.24f, 0.14f, 0.04f, 0.03f, B.strap);
        // side pouches + radio (right)
        for (int sd = -1; sd <= 1; sd += 2){
            XBOX(R, sd * 0.31f, -0.20f, -0.10f, 0.06f, 0.22f, 0.20f, B.accent);
            XBOX(R, sd * 0.33f, -0.28f, -0.10f, 0.02f, 0.04f, 0.18f, B.strap);
            if (sd > 0){
                XBOX(R, sd * 0.32f, 0.14f, -0.06f, 0.05f, 0.24f, 0.10f, B.accent);
                XBOX(R, sd * 0.34f, 0.34f, -0.06f, 0.02f, 0.20f, 0.02f, B.strap);
                XSPH(R, sd * 0.34f, 0.46f, -0.06f, 0.03f, 0.03f, 0.03f, B.accent);
            }
        }
        // bottom reinforcement
        XBOX(R, 0.0f, -0.30f, -0.14f, 0.56f, 0.04f, 0.24f, B.strap);
        // thick harness
        XBOX(R, -0.26f, 0.14f, 0.12f, 0.08f, 0.62f, 0.07f, B.strap);
        XBOX(R,  0.26f, 0.14f, 0.12f, 0.08f, 0.62f, 0.07f, B.strap);
        // sternum strap
        XBOX(R, 0.0f, -0.06f, 0.18f, 0.50f, 0.04f, 0.03f, B.strap);
    }

    else {
        // ============ PHANTOM PACK (sleek, futuristic) ============
        float pv[3] = {1.00f, 0.55f, 0.85f};
        XBOX(R, 0.0f, 0.0f, -0.14f, 0.42f, 0.62f, 0.20f, B.main);
        // angled top
        Mat4 topc = matMul(R, matT(0.0f, 0.32f, -0.14f));
        topc = matMul(topc, matRX(-0.45f));
        XBOX(topc, 0.0f, 0.0f, 0.0f, 0.40f, 0.10f, 0.20f, B.accent);
        // angled bottom
        Mat4 botc = matMul(R, matT(0.0f, -0.32f, -0.14f));
        botc = matMul(botc, matRX(0.45f));
        XBOX(botc, 0.0f, 0.0f, 0.0f, 0.40f, 0.10f, 0.20f, B.accent);
        // glowing central spine
        XBOX(R, 0.0f, 0.0f, -0.25f, 0.04f, 0.54f, 0.02f, pv);
        // horizontal accent lines
        for (int i = 0; i < 3; i++)
            XBOX(R, 0.0f, -0.16f + i*0.16f, -0.24f, 0.34f, 0.008f, 0.008f, B.accent);
        // angled side fins
        Mat4 finR = matMul(R, matT(0.22f, 0.06f, -0.14f));
        finR = matMul(finR, matRZ(-0.35f));
        XBOX(finR, 0.0f, 0.0f, 0.0f, 0.03f, 0.42f, 0.16f, B.accent);
        Mat4 finL = matMul(R, matT(-0.22f, 0.06f, -0.14f));
        finL = matMul(finL, matRZ(0.35f));
        XBOX(finL, 0.0f, 0.0f, 0.0f, 0.03f, 0.42f, 0.16f, B.accent);
        // antenna pair
        XBOX(R, -0.10f, 0.40f, -0.10f, 0.02f, 0.16f, 0.02f, B.accent);
        XBOX(R,  0.10f, 0.40f, -0.10f, 0.02f, 0.16f, 0.02f, B.accent);
        XSPH(R, -0.10f, 0.50f, -0.10f, 0.03f, 0.03f, 0.03f, pv);
        XSPH(R,  0.10f, 0.50f, -0.10f, 0.03f, 0.03f, 0.03f, pv);
        // slim harness
        XBOX(R, -0.20f, 0.14f, 0.10f, 0.06f, 0.60f, 0.05f, B.strap);
        XBOX(R,  0.20f, 0.14f, 0.10f, 0.06f, 0.60f, 0.05f, B.strap);
        XSPH(R, -0.20f, -0.14f, 0.12f, 0.04f, 0.04f, 0.04f, pv);
        XSPH(R,  0.20f, -0.14f, 0.12f, 0.04f, 0.04f, 0.04f, pv);
    }
}


// ================================================================
//  GUN — held in right hand
// ================================================================
static void drawGun(const Mat4& root, float breathe, float sway){
    if (gGunIdx < 0) return;
    // Gun held in front of chest, both hands supporting
    Mat4 G = matMul(root, matT(0.10f, 1.16f + breathe*0.35f, 0.55f));
    G = matMul(G, matRX(-0.08f));

    float body[3], accent[3], dark[3];
    if (gGunIdx == 0){        body[0]=0.37f;body[1]=0.78f;body[2]=1.00f;
                              accent[0]=0.20f;accent[1]=0.55f;accent[2]=0.85f;
                              dark[0]=0.10f;dark[1]=0.15f;dark[2]=0.22f; }
    else if (gGunIdx == 1){   body[0]=0.25f;body[1]=0.66f;body[2]=1.00f;
                              accent[0]=0.10f;accent[1]=0.42f;accent[2]=0.75f;
                              dark[0]=0.10f;dark[1]=0.15f;dark[2]=0.22f; }
    else if (gGunIdx == 2){   body[0]=0.69f;body[1]=0.52f;body[2]=1.00f;
                              accent[0]=0.42f;accent[1]=0.22f;accent[2]=0.80f;
                              dark[0]=0.16f;dark[1]=0.10f;dark[2]=0.28f; }
    else if (gGunIdx == 3){   body[0]=0.66f;body[1]=0.44f;body[2]=1.00f;
                              accent[0]=0.42f;accent[1]=0.22f;accent[2]=0.80f;
                              dark[0]=0.16f;dark[1]=0.10f;dark[2]=0.28f; }
    else {                    body[0]=1.00f;body[1]=0.78f;body[2]=0.34f;
                              accent[0]=0.85f;accent[1]=0.55f;accent[2]=0.15f;
                              dark[0]=0.30f;dark[1]=0.20f;dark[2]=0.06f; }

    if (gGunIdx == 0){
        XBOX(G, 0.0f,  0.00f,  0.20f, 0.09f, 0.11f, 0.52f, body);
        XBOX(G, 0.0f, -0.02f, -0.10f, 0.08f, 0.10f, 0.14f, dark);
        XBOX(G, 0.0f, -0.12f,  0.10f, 0.07f, 0.16f, 0.10f, dark);
        XBOX(G, 0.0f, -0.05f,  0.02f, 0.07f, 0.09f, 0.10f, dark);
        XBOX(G, 0.0f,  0.00f, -0.20f, 0.07f, 0.11f, 0.16f, dark);
        XBOX(G, 0.0f,  0.01f,  0.55f, 0.045f, 0.045f, 0.28f, accent);
        XBOX(G, 0.0f,  0.11f,  0.20f, 0.055f, 0.055f, 0.16f, accent);
        XBOX(G, 0.0f,  0.07f,  0.55f, 0.06f, 0.04f, 0.10f, dark);
        XBOX(G, 0.0f, -0.06f,  0.30f, 0.06f, 0.10f, 0.08f, dark);
    }
    else if (gGunIdx == 1){
        XBOX(G, 0.0f,  0.00f,  0.14f, 0.09f, 0.11f, 0.34f, body);
        XBOX(G, 0.0f, -0.15f,  0.12f, 0.06f, 0.18f, 0.08f, dark);
        XBOX(G, 0.0f, -0.06f,  0.00f, 0.07f, 0.09f, 0.09f, dark);
        XBOX(G, 0.0f,  0.00f, -0.12f, 0.07f, 0.09f, 0.09f, dark);
        XBOX(G, 0.0f,  0.01f,  0.42f, 0.045f, 0.045f, 0.18f, accent);
        XBOX(G, 0.0f,  0.09f,  0.10f, 0.045f, 0.045f, 0.12f, accent);
        XBOX(G, 0.0f, -0.04f,  0.24f, 0.06f, 0.08f, 0.08f, dark);
    }
    else if (gGunIdx == 2){
        XBOX(G, 0.0f,  0.00f,  0.22f, 0.07f, 0.11f, 0.58f, body);
        XBOX(G, 0.0f, -0.05f,  0.00f, 0.06f, 0.08f, 0.12f, dark);
        XBOX(G, 0.0f,  0.00f, -0.22f, 0.07f, 0.11f, 0.22f, dark);
        XBOX(G, 0.0f,  0.02f,  0.75f, 0.035f, 0.035f, 0.42f, accent);
        XBOX(G, 0.0f,  0.14f,  0.20f, 0.055f, 0.08f, 0.22f, accent);
        XBOX(G, 0.0f,  0.12f,  0.10f, 0.055f, 0.055f, 0.07f, dark);
        XBOX(G, 0.0f, -0.11f,  0.10f, 0.055f, 0.11f, 0.09f, dark);
        XBOX(G, 0.0f,  0.00f,  0.50f, 0.055f, 0.06f, 0.14f, dark);
    }
    else if (gGunIdx == 3){
        XBOX(G, 0.0f,  0.00f,  0.18f, 0.10f, 0.11f, 0.44f, body);
        XBOX(G, 0.0f,  0.05f,  0.55f, 0.055f, 0.055f, 0.30f, accent);
        XBOX(G, 0.0f, -0.05f,  0.55f, 0.055f, 0.055f, 0.30f, accent);
        XBOX(G, 0.0f,  0.00f, -0.06f, 0.09f, 0.09f, 0.16f, dark);
        XBOX(G, 0.0f, -0.05f,  0.00f, 0.07f, 0.09f, 0.11f, dark);
        XBOX(G, 0.0f,  0.00f, -0.22f, 0.07f, 0.11f, 0.18f, dark);
        XBOX(G, 0.0f, -0.02f,  0.28f, 0.055f, 0.07f, 0.15f, dark);
    }
    else {
        XBOX(G, 0.0f,  0.00f,  0.10f, 0.07f, 0.11f, 0.26f, body);
        XBOX(G, 0.0f, -0.09f, -0.03f, 0.055f, 0.15f, 0.08f, dark);
        XBOX(G, 0.0f,  0.01f,  0.30f, 0.035f, 0.035f, 0.12f, accent);
        XBOX(G, 0.0f,  0.07f,  0.08f, 0.045f, 0.035f, 0.07f, accent);
    }
}

static void drawPet(float t){
    if (gPetIdx <= 0) return;
    PetColors& P = gPets[gPetIdx];
    int kind = gPetIdx;
    float bob  = sinf(t*2.4f) * 0.04f;
    float sway = sinf(t*1.3f) * 0.22f;
    Mat4 base = matMul(matT(0.62f, 0.36f + bob, 0.20f), matRY(gPetYaw + sway));
    drawSphere(matMul(base, matS(0.26f, 0.22f, 0.30f)), P.body);
    drawSphere(matMul(base, matMul(matT(0.0f, 0.16f, 0.04f), matS(0.22f, 0.22f, 0.22f))), P.body);
    drawBox(matMul(base, matMul(matT(-0.055f, 0.18f, 0.15f), matS(0.04f, 0.04f, 0.04f))), P.eye);
    drawBox(matMul(base, matMul(matT( 0.055f, 0.18f, 0.15f), matS(0.04f, 0.04f, 0.04f))), P.eye);
    if (kind == 1){
        drawSphere(matMul(base, matMul(matT(-0.16f, 0.04f, 0.0f), matS(0.10f, 0.18f, 0.24f))), P.accent);
        drawSphere(matMul(base, matMul(matT( 0.16f, 0.04f, 0.0f), matS(0.10f, 0.18f, 0.24f))), P.accent);
        drawSphere(matMul(base, matMul(matT(0.0f, 0.15f, 0.16f), matS(0.07f, 0.06f, 0.09f))), P.accent);
    } else if (kind == 2){
        drawSphere(matMul(base, matMul(matT(-0.06f, 0.28f, 0.02f), matS(0.07f, 0.10f, 0.07f))), P.accent);
        drawSphere(matMul(base, matMul(matT( 0.06f, 0.28f, 0.02f), matS(0.07f, 0.10f, 0.07f))), P.accent);
        drawSphere(matMul(base, matMul(matT(0.0f, 0.14f, 0.16f), matS(0.10f, 0.07f, 0.09f))), P.accent);
    } else if (kind == 3){
        drawSphere(matMul(base, matMul(matT(-0.18f, 0.08f, -0.02f), matS(0.09f, 0.24f, 0.18f))), P.accent);
        drawSphere(matMul(base, matMul(matT( 0.18f, 0.08f, -0.02f), matS(0.09f, 0.24f, 0.18f))), P.accent);
        drawSphere(matMul(base, matMul(matT(0.0f, 0.02f, -0.18f), matS(0.11f, 0.09f, 0.16f))), P.accent);
    } else if (kind == 4){
        drawSphere(matMul(base, matMul(matT(-0.06f, 0.29f, 0.02f), matS(0.07f, 0.11f, 0.07f))), P.accent);
        drawSphere(matMul(base, matMul(matT( 0.06f, 0.29f, 0.02f), matS(0.07f, 0.11f, 0.07f))), P.accent);
        drawSphere(matMul(base, matMul(matT(0.0f, 0.06f, -0.16f), matS(0.06f, 0.18f, 0.06f))), P.accent);
    } else if (kind == 5){
        drawSphere(matMul(base, matMul(matT(-0.16f, 0.06f, 0.0f), matS(0.07f, 0.24f, 0.22f))), P.accent);
        drawSphere(matMul(base, matMul(matT( 0.16f, 0.06f, 0.0f), matS(0.07f, 0.24f, 0.22f))), P.accent);
        float glow[3] = {0.35f, 0.95f, 1.00f};
        drawBox(matMul(base, matMul(matT(-0.055f, 0.18f, 0.15f), matS(0.05f, 0.05f, 0.05f))), glow);
        drawBox(matMul(base, matMul(matT( 0.055f, 0.18f, 0.15f), matS(0.05f, 0.05f, 0.05f))), glow);
    }
}

// ================================================================
//  RENDER LOOP
// ================================================================
static void frame(){
    double now=emscripten_get_now();
    if(gLast==0) gLast=now;
    gT += (now-gLast)/1000.0;
    gLast = now;
    float t = (float)gT;

    int w=0,h=0;
    emscripten_get_canvas_element_size("#canvas",&w,&h);
    if(w<=0||h<=0) return;
    glViewport(0,0,w,h);
    glClearColor(0.04f,0.05f,0.09f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    float asp=(float)w/(float)h;
    float camX = 1.6f, camY = 2.2f, camZ = 5.4f;
    if (asp < 0.8f){ camX *= 1.2f; camY = 2.6f; camZ = 7.0f; }
    Mat4 proj = matPersp(gFOV*3.14159265f/180.0f, asp, 0.1f, 100.0f);
    Mat4 view = matLookAt(camX, camY, camZ, 0.0f, 1.15f, 0.0f);
    gVP = matMul(proj, view);

    drawBoxRGB(matMul(matT(0,-0.08f,0), matS(6.0f,0.15f,6.0f)), 0.10f,0.13f,0.20f);
    {
        const int N=32; const float rr=1.9f;
        float spin = t*0.6f;
        for(int i=0;i<N;i++){
            float a=(i/(float)N)*6.2831853f + spin;
            float x=cosf(a)*rr, z=sinf(a)*rr;
            float p=0.5f+0.5f*sinf(t*3.0f + i*0.4f);
            Mat4 m = matMul(matT(x,0.05f,z), matS(0.10f, 0.06f+0.03f*p, 0.10f));
            drawBoxRGB(m, 0.05f, 0.7f+0.3f*p, 1.0f);
        }
    }

    Palette& P = gPalettes[gPaletteIdx];

    // Natural breathing (~4.5s cycle), body lags behind
    float bP = t * 1.35f;
    float breathe = sinf(bP) * 0.055f;
    float bob     = sinf(bP + 0.55f) * 0.014f;
    float sway    = sinf(t * 0.85f) * 0.042f;

    if (gAutoRotate){
        float dt = 0.016f;
        gCharYaw += 0.35f * dt;
    }
    Mat4 root = matMul(matRY(gCharYaw), matT(0, bob, 0));

    // ---- dispatch per character ----
    switch(gPaletteIdx){
        case 0: drawAlpha (root, P, breathe, sway); break;
        case 1: drawNova  (root, P, breathe, sway); break;
        case 2: drawGhost (root, P, breathe, sway); break;
        case 3: drawBlaze (root, P, breathe, sway); break;
        case 4: drawShadow(root, P, breathe, sway); break;
        default: drawAlpha(root, P, breathe, sway); break;
    }

    drawBackpack(root);
    drawGun(root, breathe, sway);
    drawPet(t);
}

int main(){
    EmscriptenWebGLContextAttributes attr;
    emscripten_webgl_init_context_attributes(&attr);
    attr.majorVersion = 1; attr.minorVersion = 0;
    attr.antialias = EM_TRUE; attr.depth = EM_TRUE; attr.alpha = EM_FALSE;
    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx = emscripten_webgl_create_context("#canvas", &attr);
    if(ctx<=0){ printf("WebGL context failed\n"); return 1; }
    emscripten_webgl_make_context_current(ctx);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE); glCullFace(GL_BACK);

    GLuint vs=sh(GL_VERTEX_SHADER,VERT);
    GLuint fs=sh(GL_FRAGMENT_SHADER,FRAG);
    gProg=glCreateProgram();
    glAttachShader(gProg,vs); glAttachShader(gProg,fs);
    glLinkProgram(gProg); glUseProgram(gProg);

    aPos    = glGetAttribLocation(gProg,"aPos");
    aNormal = glGetAttribLocation(gProg,"aNormal");
    uMVP    = glGetUniformLocation(gProg,"uMVP");
    uModel  = glGetUniformLocation(gProg,"uModel");
    uColor  = glGetUniformLocation(gProg,"uColor");

    glGenBuffers(1,&gVBO);
    glBindBuffer(GL_ARRAY_BUFFER,gVBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(CUBE),CUBE,GL_STATIC_DRAW);
    glGenBuffers(1,&gIBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,gIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(IDX),IDX,GL_STATIC_DRAW);

    buildSphere();

    emscripten_set_mousedown_callback("#canvas", nullptr, EM_TRUE, on_mouse_down);
    emscripten_set_mouseup_callback  ("#canvas", nullptr, EM_TRUE, on_mouse_up);
    emscripten_set_mousemove_callback("#canvas", nullptr, EM_TRUE, on_mouse_move);
    emscripten_set_touchstart_callback("#canvas", nullptr, EM_TRUE, on_touch_start);
    emscripten_set_touchend_callback  ("#canvas", nullptr, EM_TRUE, on_touch_end);
    emscripten_set_touchcancel_callback("#canvas", nullptr, EM_TRUE, on_touch_end);
    emscripten_set_touchmove_callback ("#canvas", nullptr, EM_TRUE, on_touch_move);

    emscripten_set_main_loop(frame,0,1);
    return 0;
}
