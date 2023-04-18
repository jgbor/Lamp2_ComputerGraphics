#include "framework.h"

//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
//---------------------------
    float f; // function value
    T d;  // derivatives
    Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
    Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
    Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
    Dnum operator*(Dnum r) {
        return Dnum(f * r.f, f * r.d + d * r.f);
    }
    Dnum operator/(Dnum r) {
        return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
    }
};

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f)*g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f)*g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f)*g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f)*g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f)*g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
    return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 40;

//---------------------------
struct Camera { // 3D camera
//---------------------------
    vec3 wEye, wLookat, wVup;   // extrinsic
    float fov, asp, fp, bp;		// intrinsic
public:
    Camera() {
        asp = (float)windowWidth / windowHeight;
        fov = 75.0f * (float)M_PI / 180.0f;
        fp = 1; bp = 20;
    }
    mat4 V() { // view matrix: translates the center to the origin
        vec3 w = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wVup, w));
        vec3 v = cross(w, u);
        return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
                                                   u.y, v.y, w.y, 0,
                                                   u.z, v.z, w.z, 0,
                                                   0,   0,   0,   1);
    }

    mat4 P() { // projection matrix
        return mat4(1 / (tan(fov / 2)*asp), 0,                0,                      0,
                    0,                      1 / tan(fov / 2), 0,                      0,
                    0,                      0,                -(fp + bp) / (bp - fp), -1,
                    0,                      0,                -2 * fp*bp / (bp - fp),  0);
    }

    void Animate(float dt) {
        wEye = vec3((wEye.x - wLookat.x) * cos(dt) + (wEye.y - wLookat.y) * sin(dt) + wLookat.x,
                   -(wEye.x - wLookat.x) * sin(dt) + (wEye.y - wLookat.y) * cos(dt) + wLookat.y,
                   wEye.z);
    }
};

//---------------------------
struct Material {
//---------------------------
    vec3 kd, ks, ka;
    float shininess;
};

//---------------------------
struct Light {
//---------------------------
    vec3 La, Le;
    vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};

class LampTexture : public Texture {
public:
    LampTexture(const int width, const int height) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 color(0.6f, 0.6f, 0.2f, 1);
        for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
                image[y * width + x] = color;
            }
        create(width, height, image, GL_NEAREST);
    }
};

class PlaneTexture : public Texture {
public:
    PlaneTexture(const int width, const int height) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 color(0.8f, 1.0f, 1.0f, 1);
        for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
                image[y * width + x] = color;
            }
        create(width, height, image, GL_NEAREST);
    }
};

class ShadowTexture : public Texture {
public:
    ShadowTexture(const int width, const int height) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 color(0, 0, 0, 0);
        for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
                image[y * width + x] = color;
            }
        create(width, height, image, GL_NEAREST);
    }
};

struct RenderState {
    mat4	           MVP, M, Minv, V, P;
    Material *         material;
    std::vector<Light> lights;
    Texture *          texture;
    vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
//---------------------------
public:
    virtual void Bind(RenderState state) = 0;

    void setUniformMaterial(const Material& material, const std::string& name) {
        setUniform(material.kd, name + ".kd");
        setUniform(material.ks, name + ".ks");
        setUniform(material.ka, name + ".ka");
        setUniform(material.shininess, name + ".shininess");
    }

    void setUniformLight(const Light& light, const std::string& name) {
        setUniform(light.La, name + ".La");
        setUniform(light.Le, name + ".Le");
        setUniform(light.wLightPos, name + ".wLightPos");
    }
};

//---------------------------
class PhongShader : public Shader {
//---------------------------
    const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

    // fragment shader in GLSL
    const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;

        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La +
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
    PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use(); 		// make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniform(*state.texture, std::string("diffuseTexture"));
        setUniformMaterial(*state.material, "material");

        setUniform((int)state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};

class ShadowShader : public Shader {
    const char * vertexSource = R"(
        #version 330
		precision highp float;

        uniform mat4 MVP;
        layout(location = 0) in vec3 vtxPos;
        void main() { gl_Position = vec4(vtxPos, 1) * MVP; }
)";

    const char * fragmentsSource = R"(
        #version 330
		precision highp float;

        out vec4 fragmentColor;
	    void main() { fragmentColor = vec4(0, 0, 0, 1); }
)";
public:
    ShadowShader() {
        create(vertexSource, fragmentsSource, "fragmentColor");
    }

    void Bind(RenderState state) {
        Use();
        mat4 MVP = state.M * state.V * state.P;
        setUniform(MVP, "MVP");
    }
};


//---------------------------
class Geometry {
//---------------------------
protected:
    unsigned int vao, vbo;        // vertex array object
public:
    Geometry() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }
    virtual void Draw() = 0;
    ~Geometry() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
};

//---------------------------
class ParamSurface : public Geometry {
//---------------------------
    struct VertexData {
        vec3 position, normal;
        vec2 texcoord;
    };

    unsigned int nVtxPerStrip, nStrips;
public:
    ParamSurface() { nVtxPerStrip = nStrips = 0; }

    virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

    VertexData GenVertexData(float u, float v) {
        VertexData vtxData;
        vtxData.texcoord = vec2(u, v);
        Dnum2 X, Y, Z;
        Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
        eval(U, V, X, Y, Z);
        vtxData.position = vec3(X.f, Y.f, Z.f);
        vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
        vtxData.normal = cross(drdU, drdV);
        return vtxData;
    }

    void create(int N = tessellationLevel, int M = tessellationLevel) {
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        std::vector<VertexData> vtxData;	// vertices on the CPU
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= M; j++) {
                vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
                vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
            }
        }
        glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
    }

    void Draw() {
        glBindVertexArray(vao);
        for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
    }
};

//---------------------------
class Sphere : public ParamSurface {
//---------------------------
public:
    Sphere() { create(); }
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
        U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
        X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
    }
};

class Paraboloid : public ParamSurface {
    float height = 1.0f;
public:
    Paraboloid(){ create(); }
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
        U = U * 2.0f * (float)M_PI, V = V * height;
        X = Cos(U) * V; Y = Sin(U) * V; Z = V*V;
    }
};

class Plane : public ParamSurface {
public:
    Plane(){ create(); }
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
        X = U; Y = V; Z = 0;
    }
};

//---------------------------
class Cylinder : public ParamSurface {
//---------------------------
public:
    Cylinder() { create(); }
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
        U = U * 2.0f * M_PI, V = V * 2;
        X = Cos(U); Y = Sin(U); Z = V;
    }
};

//---------------------------
struct Object {
//---------------------------
    Shader *   shader;
    Material * material;
    Texture *  texture;
    Geometry * geometry;
    vec3 scale, translation, rotAxis;
    float rotAngle;
    std::vector<Object *> children;
    bool shouldRot;

public:
    Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
            scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotAxis(0, 1, 0), rotAngle(0),shouldRot(true) {
        shader = _shader;
        texture = _texture;
        material = _material;
        geometry = _geometry;
    }

    void addChild(Object * child){
        children.push_back(child);
    }

    void Draw(RenderState state) {
        state.M = ScaleMatrix(scale) * RotationMatrix(rotAngle, rotAxis) * TranslateMatrix(translation) * state.M;
        state.Minv = state.Minv * TranslateMatrix(-translation) * RotationMatrix(-rotAngle, rotAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));

        state.MVP = state.M * state.V * state.P;
        state.material = material;
        state.texture = texture;
        shader->Bind(state);
        geometry->Draw();
        for (Object * child : children) child->Draw(state);
    }

    virtual void Animate(float tstart, float tend) {
        if(shouldRot)
            rotAngle = 0.9f * sin(tend);
        for (Object * child : children) child->Animate(tstart,tend);
    }
};

mat4 UnixMatrix(){
    return mat4(1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1);
}

//---------------------------
class Scene {
//---------------------------
    std::vector<Object *> objects;
    Camera camera; // 3D camera
    std::vector<Light> lights;
public:
    void lampa(Shader * shader, Material * material, Texture * lampTexture){
        Geometry * sphere = new Sphere();
        Geometry * paraboloid = new Paraboloid();
        Geometry * cylinder = new Cylinder();

        Object * talp = new Object(shader, material, lampTexture, cylinder);
        talp->translation = vec3(0, 0, -2);
        talp->scale = vec3(2, 2, 0.25f);
        talp->shouldRot=false;
        objects.push_back(talp);

        Object * teto = new Object(shader, material, lampTexture, sphere);
        teto->translation = vec3(0, 0, 1.75f);
        teto->scale = vec3(1, 1, 0.000001f);
        teto->shouldRot=false;
        talp->addChild(teto);

        Object * alsocsuklo = new Object(shader, material, lampTexture, sphere);
        alsocsuklo->translation = vec3(0, 0, 1);
        alsocsuklo->scale = vec3(0.25f, 0.25f, 2);
        alsocsuklo->rotAxis = vec3(0,0,1);
        talp->addChild(alsocsuklo);

        Object * rud1 = new Object(shader, material, lampTexture, cylinder);
        rud1->translation = vec3(0, 0, 0.5f);
        rud1->scale = vec3(0.5f, 0.5f, 2);
        alsocsuklo->addChild(rud1);

        Object * kozepsocsuklo = new Object(shader, material, lampTexture, sphere);
        kozepsocsuklo->translation = vec3(0, 0, 2);
        kozepsocsuklo->scale = vec3(2, 2, 0.5f);
        kozepsocsuklo->rotAxis = vec3(0,0,1);
        rud1->addChild(kozepsocsuklo);

        Object * rud2 = new Object(shader, material, lampTexture, cylinder);
        rud2->translation = vec3(0, 0, 0.5f);
        rud2->scale = vec3(0.5f, 0.5f, 2);
        kozepsocsuklo->addChild(rud2);

        Object * felsocsuklo = new Object(shader, material, lampTexture, sphere);
        felsocsuklo->translation = vec3(0, 0, 2);
        felsocsuklo->scale = vec3(2, 2, 0.5f);
        felsocsuklo->rotAxis = vec3(0,0,1);
        rud2->addChild(felsocsuklo);

        Object * bura = new Object(shader, material, lampTexture, paraboloid);
        bura->translation = vec3(0, 0, 0.75f);
        bura->scale = vec3(3, 3, 2);
        bura->rotAxis = vec3(0, 0, 1);
        felsocsuklo->addChild(bura);
    }

    void Build() {
        // Shaders
        Shader * phongShader = new PhongShader();
        Shader * shadowShader = new ShadowShader();

        // Materials
        Material * material0 = new Material;
        material0->kd = vec3(0.6f, 0.4f, 0.2f);
        material0->ks = vec3(1, 1, 1);
        material0->ka = vec3(0.5f, 0.5f, 0.5f);
        material0->shininess = 45;

        Material * material1 = new Material;
        material1->kd = vec3(0.8f, 0.6f, 0.4f);
        material1->ks = vec3(0.3f, 0.3f, 0.3f);
        material1->ka = vec3(0.2f, 0.2f, 0.2f);
        material1->shininess = 100;

        // Textures
        Texture * lampTexture = new LampTexture(4, 8);
        Texture * planeTexture = new PlaneTexture(64,64);
        Texture * shadowTexture = new ShadowTexture(4,8);

        // Geometries
        Geometry * plane = new Plane();

        // Create objects by setting up their vertex data on the GPU
        Object * talaj = new Object(phongShader, material1, planeTexture, plane);
        talaj->translation = vec3(-50,-50,-2);
        talaj->scale = vec3(5000,5000,1);
        talaj->shouldRot=false;
        objects.push_back(talaj);

        lampa(phongShader,material0,lampTexture);
        lampa(shadowShader,material0,shadowTexture);


        // Camera
        camera.wEye = vec3(0, 8, 0);
        camera.wLookat = vec3(0, 0, 0);
        camera.wVup = vec3(0, 0, 1);

        // Lights
        lights.resize(2);
        lights[0].wLightPos = vec4(5, 5, 4, 0);	// ideal point -> directional light source
        lights[0].La = vec3(0.1f, 0.1f, 0.1f);
        lights[0].Le = vec3(1, 1, 1);

        lights[1].wLightPos = vec4(5, 5, 4, 0);	// ideal point -> directional light source
        lights[1].La = vec3(0.2f, 0.2f, 0.2f);
        lights[1].Le = vec3(1, 1, 1);
    }

    void Render() {
        RenderState state;
        state.M = state.Minv = UnixMatrix();
        state.wEye = camera.wEye;
        state.V = camera.V();
        state.P = camera.P();
        state.lights = lights;
        for (Object * obj : objects) obj->Draw(state);
    }

    void Animate(float tstart, float tend) {
        camera.Animate(0.01f);
        for (Object * obj : objects) obj->Animate(tstart, tend);
    }
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    scene.Render();
    glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { }

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    static float tend = 0;
    const float dt = 0.1f; // dt is ?infinitesimal?
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

    for (float t = tstart; t < tend; t += dt) {
        float Dt = fmin(dt, tend - t);
        scene.Animate(t, t + Dt);
    }
    glutPostRedisplay();
}