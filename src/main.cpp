#include <emscripten.h>
#include <emscripten/html5.h>
#include <GLES2/gl2.h>
#include <cmath>
#include <cstring>
#include <cstdio>

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

// 0 = character is active subject, 1 = pet is active subject
static int gActiveSubject = 0;

extern "C" EMSCRIPTEN_KEEPALIVE
void set_character_preset(int idx){ if(idx<0)idx=0; if(idx>4)idx=4; gPaletteIdx=idx; }

extern "C" EMSCRIPTEN_KEEPALIVE
void set_pet_preset(int idx){ if(idx<0)idx=0; if(idx>5)idx=5; gPetIdx=idx; }

extern "C" EMSCRIPTEN_KEEPALIVE
void set_active_subject(int s){ gActiveSubject = (s==1) ? 1 : 0; }

// Independent yaw for character and pet
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
    float delta = -(float)dx * 0.010f;
    if (gActiveSubject == 0) gCharYaw += delta;
    else                     gPetYaw += delta * 0.5f;
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
    float delta = -(float)dx * 0.012f;
    if (gActiveSubject == 0) gCharYaw += delta;
    else                     gPetYaw += delta * 0.5f;
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
static void drawBox(const Mat4&m,const float c[3]){
    Mat4 mvp=matMul(gVP,m);
    glUniformMatrix4fv(uModel,1,GL_FALSE,m.m);
    glUniformMatrix4fv(uMVP,1,GL_FALSE,mvp.m);
    glUniform3f(uColor,c[0],c[1],c[2]);
    glDrawElements(GL_TRIANGLES,36,GL_UNSIGNED_SHORT,0);
}
static void drawBoxRGB(const Mat4&m,float r,float g,float b){
    float c[3]={r,g,b};
    drawBox(m,c);
}

static void drawPet(float t){
    if (gPetIdx <= 0) return;
    PetColors& P = gPets[gPetIdx];
    int kind = gPetIdx;

    float bob  = sinf(t*2.4f) * 0.04f;
    float sway = sinf(t*1.3f) * 0.22f;

    // Pet has its own yaw + fixed world position next to character
    Mat4 base = matMul(matT(0.62f, 0.36f + bob, 0.20f),
                       matRY(gPetYaw + sway));

    drawBox(matMul(base, matS(0.22f, 0.20f, 0.26f)), P.body);
    drawBox(matMul(base, matMul(matT(0.0f, 0.16f, 0.04f), matS(0.20f, 0.20f, 0.20f))), P.body);
    drawBox(matMul(base, matMul(matT(-0.055f, 0.18f, 0.15f), matS(0.04f, 0.04f, 0.04f))), P.eye);
    drawBox(matMul(base, matMul(matT( 0.055f, 0.18f, 0.15f), matS(0.04f, 0.04f, 0.04f))), P.eye);

    if (kind == 1){
        drawBox(matMul(base, matMul(matT(-0.16f, 0.04f, 0.0f), matS(0.09f, 0.16f, 0.22f))), P.accent);
        drawBox(matMul(base, matMul(matT( 0.16f, 0.04f, 0.0f), matS(0.09f, 0.16f, 0.22f))), P.accent);
        drawBox(matMul(base, matMul(matT(0.0f, 0.15f, 0.16f), matS(0.06f, 0.05f, 0.08f))), P.accent);
    } else if (kind == 2){
        drawBox(matMul(base, matMul(matT(-0.06f, 0.28f, 0.02f), matS(0.06f, 0.09f, 0.06f))), P.accent);
        drawBox(matMul(base, matMul(matT( 0.06f, 0.28f, 0.02f), matS(0.06f, 0.09f, 0.06f))), P.accent);
        drawBox(matMul(base, matMul(matT(0.0f, 0.14f, 0.16f), matS(0.09f, 0.06f, 0.08f))), P.accent);
    } else if (kind == 3){
        drawBox(matMul(base, matMul(matT(-0.18f, 0.08f, -0.02f), matS(0.08f, 0.22f, 0.16f))), P.accent);
        drawBox(matMul(base, matMul(matT( 0.18f, 0.08f, -0.02f), matS(0.08f, 0.22f, 0.16f))), P.accent);
        drawBox(matMul(base, matMul(matT(0.0f, 0.02f, -0.18f), matS(0.10f, 0.08f, 0.14f))), P.accent);
    } else if (kind == 4){
        drawBox(matMul(base, matMul(matT(-0.06f, 0.29f, 0.02f), matS(0.06f, 0.10f, 0.06f))), P.accent);
        drawBox(matMul(base, matMul(matT( 0.06f, 0.29f, 0.02f), matS(0.06f, 0.10f, 0.06f))), P.accent);
        drawBox(matMul(base, matMul(matT(0.0f, 0.06f, -0.16f), matS(0.05f, 0.16f, 0.05f))), P.accent);
    } else if (kind == 5){
        drawBox(matMul(base, matMul(matT(-0.16f, 0.06f, 0.0f), matS(0.06f, 0.22f, 0.20f))), P.accent);
        drawBox(matMul(base, matMul(matT( 0.16f, 0.06f, 0.0f), matS(0.06f, 0.22f, 0.20f))), P.accent);
        float glow[3] = {0.35f, 0.95f, 1.00f};
        drawBox(matMul(base, matMul(matT(-0.055f, 0.18f, 0.15f), matS(0.05f, 0.05f, 0.05f))), glow);
        drawBox(matMul(base, matMul(matT( 0.055f, 0.18f, 0.15f), matS(0.05f, 0.05f, 0.05f))), glow);
    }
}

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
    Mat4 proj = matPersp(45.0f*3.14159265f/180.0f, asp, 0.1f, 100.0f);
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
    float skin[3] = {1.00f,0.82f,0.62f};

    float breathe = sinf(t*1.6f)*0.04f;
    float bob     = sinf(t*1.6f)*0.02f;
    float sway    = sinf(t*1.2f)*0.05f;

    Mat4 root = matMul(matRY(gCharYaw), matT(0, bob, 0));

    drawBox(matMul(root, matMul(matT(-0.16f,0.40f,0), matS(0.20f,0.80f,0.20f))), P.legs);
    drawBox(matMul(root, matMul(matT( 0.16f,0.40f,0), matS(0.20f,0.80f,0.20f))), P.legs);
    float boot[3]={0.05f,0.06f,0.08f};
    drawBox(matMul(root, matMul(matT(-0.16f,0.06f,0.04f), matS(0.24f,0.14f,0.28f))), boot);
    drawBox(matMul(root, matMul(matT( 0.16f,0.06f,0.04f), matS(0.24f,0.14f,0.28f))), boot);

    drawBox(matMul(root, matMul(matT(0.0f, 1.20f + breathe*0.5f, 0), matS(0.62f + breathe,0.82f,0.34f))), P.torso);
    drawBox(matMul(root, matMul(matT(0.0f, 1.28f + breathe*0.5f, 0.18f), matS(0.44f,0.30f,0.06f))), P.chest);
    drawBox(matMul(root, matMul(matT(0.0f, 0.86f, 0.0f), matS(0.64f,0.08f,0.36f))), P.belt);

    drawBox(matMul(root, matMul(matT(0.0f, 1.90f + breathe, 0), matS(0.44f,0.44f,0.44f))), skin);
    drawBox(matMul(root, matMul(matT(0.0f, 2.06f + breathe, 0), matS(0.52f,0.20f,0.52f))), P.helmet);
    drawBoxRGB(matMul(root, matMul(matT(0.0f, 1.96f + breathe, 0.24f), matS(0.42f,0.08f,0.06f))), 0.05f,0.85f,1.00f);
    float eye[3]={0.05f,0.05f,0.09f};
    drawBox(matMul(root, matMul(matT(-0.10f, 1.88f + breathe, 0.235f), matS(0.08f,0.08f,0.05f))), eye);
    drawBox(matMul(root, matMul(matT( 0.10f, 1.88f + breathe, 0.235f), matS(0.08f,0.08f,0.05f))), eye);

    float thL = -(0.12f + sway);
    float thR =  (0.12f + sway);
    Mat4 armBaseL = matMul(root, matT(-0.46f, 1.44f + breathe*0.5f, 0));
    Mat4 armBaseR = matMul(root, matT( 0.46f, 1.44f + breathe*0.5f, 0));
    Mat4 armL = matMul(armBaseL, matRZ(thL));
    Mat4 armR = matMul(armBaseR, matRZ(thR));
    drawBox(matMul(armBaseL, matS(0.24f,0.16f,0.30f)), P.chest);
    drawBox(matMul(armBaseR, matS(0.24f,0.16f,0.30f)), P.chest);
    drawBox(matMul(armL, matMul(matT(0,-0.34f,0), matS(0.16f,0.68f,0.16f))), P.torso);
    drawBox(matMul(armR, matMul(matT(0,-0.34f,0), matS(0.16f,0.68f,0.16f))), P.torso);
    drawBox(matMul(armL, matMul(matT(0,-0.72f,0), matS(0.16f,0.16f,0.16f))), skin);
    drawBox(matMul(armR, matMul(matT(0,-0.72f,0), matS(0.16f,0.16f,0.16f))), skin);

    // Pet — independent rotation
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
    glEnableVertexAttribArray(aPos);
    glVertexAttribPointer(aPos,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)0);
    glEnableVertexAttribArray(aNormal);
    glVertexAttribPointer(aNormal,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float)));

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
