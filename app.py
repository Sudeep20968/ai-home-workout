import streamlit as st
import cv2
import mediapipe as mp
import math
import time
import threading

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Home Workout",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at top left,
            #172033 0%,
            #0b0f17 40%,
            #070a10 100%
        );

    color: white;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}


/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #0b1019;
    border-right: 1px solid #202938;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    color: white !important;
}


/* HEADER */

.main-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 4px;
}

.subtitle {
    color: #8e9bad;
    font-size: 15px;
    margin-bottom: 25px;
}


/* HERO */

.hero-card {
    background:
        linear-gradient(
            135deg,
            rgba(30,41,59,0.98),
            rgba(15,23,35,0.98)
        );

    border: 1px solid #2b374b;

    border-radius: 24px;

    padding: 28px;

    margin-bottom: 24px;

    box-shadow:
        0 15px 40px rgba(0,0,0,0.30);
}

.hero-title {
    font-size: 28px;
    font-weight: 800;
}

.hero-text {
    color: #94a3b8;
    margin-top: 5px;
}


/* METRIC CARDS */

.metric-card {
    background:
        linear-gradient(
            145deg,
            rgba(25,32,46,0.95),
            rgba(13,18,27,0.95)
        );

    border: 1px solid #273143;

    border-radius: 20px;

    padding: 20px;

    min-height: 120px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.25);
}

.metric-label {
    color: #8995a8;

    font-size: 12px;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 1px;
}

.metric-value {
    color: white;

    font-size: 34px;

    font-weight: 800;

    margin-top: 8px;
}


/* FEEDBACK */

.feedback-card {
    background: #101827;

    border: 1px solid #263247;

    border-radius: 18px;

    padding: 18px 22px;

    margin-top: 18px;

    font-size: 16px;

    font-weight: 600;
}


/* BUTTON */

.stButton > button {
    height: 48px;

    border-radius: 12px;

    font-weight: 700;

    background: #151d2b;

    color: white;

    border: 1px solid #334155;
}

.stButton > button:hover {
    background: #1d2738;

    border-color: #64748b;
}


/* SELECTBOX */

div[data-baseweb="select"] > div {
    background-color: #111827;

    border-color: #334155;

    border-radius: 10px;
}


/* MOBILE */

@media only screen and (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    .main-title {
        font-size: 30px;
    }

    .subtitle {
        font-size: 13px;
    }

    .hero-card {
        padding: 20px;
    }

    .metric-card {
        min-height: 100px;
        padding: 15px;
    }

    .metric-value {
        font-size: 27px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MEDIAPIPE MODEL
# ============================================================

MODEL_PATH = "models/pose_landmarker_full.task"


@st.cache_resource
def load_model():

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    detector = vision.PoseLandmarker.create_from_options(
        options
    )

    return detector


detector = load_model()


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(
            c[1] - b[1],
            c[0] - b[0]
        )
        -
        math.atan2(
            a[1] - b[1],
            a[0] - b[0]
        )
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


def get_point(landmark, width, height):

    return (
        int(landmark.x * width),
        int(landmark.y * height)
    )


def get_form_status(score):

    if score >= 85:
        return "EXCELLENT"

    elif score >= 70:
        return "GOOD"

    else:
        return "NEEDS IMPROVEMENT"


def calorie_estimate(reps, exercise):

    if exercise == "Push-up":
        return reps * 0.35

    return reps * 0.50


# ============================================================
# SHARED WORKOUT STATE
# ============================================================

class WorkoutState:

    def __init__(self):

        self.lock = threading.Lock()

        self.reps = 0

        self.stage = "UP"

        self.angle = 0

        self.score = 0

        self.feedback = "Waiting for camera..."

        self.exercise = "Push-up"

        self.running = False

        self.start_time = None

        self.last_timestamp = 0


state = WorkoutState()


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class WorkoutProcessor(VideoProcessorBase):

    def __init__(self):

        self.local_detector = detector


    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )


        # ----------------------------------------------------
        # MIRROR CAMERA
        # ----------------------------------------------------

        image = cv2.flip(
            image,
            1
        )


        height, width, _ = image.shape


        # ----------------------------------------------------
        # BGR TO RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        timestamp = int(
            time.monotonic() * 1000
        )


        with state.lock:

            if timestamp <= state.last_timestamp:

                timestamp = (
                    state.last_timestamp + 1
                )

            state.last_timestamp = timestamp


        # ----------------------------------------------------
        # POSE DETECTION
        # ----------------------------------------------------

        result = self.local_detector.detect_for_video(
            mp_image,
            timestamp
        )


        angle = 0

        score = 100

        feedback = "Keep going! 💪"


        # ====================================================
        # PERSON DETECTED
        # ====================================================

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]


            with state.lock:

                exercise = state.exercise

                running = state.running


            # =================================================
            # PUSH-UP
            # =================================================

            if exercise == "Push-up":

                left_shoulder = get_point(
                    landmarks[11],
                    width,
                    height
                )

                left_elbow = get_point(
                    landmarks[13],
                    width,
                    height
                )

                left_wrist = get_point(
                    landmarks[15],
                    width,
                    height
                )


                right_shoulder = get_point(
                    landmarks[12],
                    width,
                    height
                )

                right_elbow = get_point(
                    landmarks[14],
                    width,
                    height
                )

                right_wrist = get_point(
                    landmarks[16],
                    width,
                    height
                )


                left_angle = calculate_angle(
                    left_shoulder,
                    left_elbow,
                    left_wrist
                )

                right_angle = calculate_angle(
                    right_shoulder,
                    right_elbow,
                    right_wrist
                )


                angle = (
                    left_angle +
                    right_angle
                ) / 2


                # REP COUNT

                if running:

                    with state.lock:

                        if angle < 90:

                            state.stage = "DOWN"


                        if (
                            angle > 160
                            and
                            state.stage == "DOWN"
                        ):

                            state.stage = "UP"

                            state.reps += 1


                # FORM SCORE

                score = 100

                feedback = "Great form! 💪"


                if (
                    state.stage == "DOWN"
                    and
                    angle > 100
                ):

                    score -= 20

                    feedback = "Go lower ⬇️"


                arm_difference = abs(
                    left_angle -
                    right_angle
                )


                if arm_difference > 25:

                    score -= 10

                    feedback = (
                        "Keep both arms balanced ⚖️"
                    )


                points = [

                    left_shoulder,
                    left_elbow,
                    left_wrist,

                    right_shoulder,
                    right_elbow,
                    right_wrist

                ]


                angle_name = "Elbow"


            # =================================================
            # SQUAT
            # =================================================

            else:

                left_hip = get_point(
                    landmarks[23],
                    width,
                    height
                )

                left_knee = get_point(
                    landmarks[25],
                    width,
                    height
                )

                left_ankle = get_point(
                    landmarks[27],
                    width,
                    height
                )


                right_hip = get_point(
                    landmarks[24],
                    width,
                    height
                )

                right_knee = get_point(
                    landmarks[26],
                    width,
                    height
                )

                right_ankle = get_point(
                    landmarks[28],
                    width,
                    height
                )


                left_angle = calculate_angle(
                    left_hip,
                    left_knee,
                    left_ankle
                )

                right_angle = calculate_angle(
                    right_hip,
                    right_knee,
                    right_ankle
                )


                angle = (
                    left_angle +
                    right_angle
                ) / 2


                # REP COUNT

                if running:

                    with state.lock:

                        if angle < 110:

                            state.stage = "DOWN"


                        if (
                            angle > 160
                            and
                            state.stage == "DOWN"
                        ):

                            state.stage = "UP"

                            state.reps += 1


                # FORM SCORE

                score = 100

                feedback = "Great squat! 🏋️"


                if (
                    state.stage == "DOWN"
                    and
                    angle > 110
                ):

                    score -= 20

                    feedback = "Go lower ⬇️"


                knee_difference = abs(
                    left_angle -
                    right_angle
                )


                if knee_difference > 20:

                    score -= 10

                    feedback = (
                        "Keep both knees balanced ⚖️"
                    )


                points = [

                    left_hip,
                    left_knee,
                    left_ankle,

                    right_hip,
                    right_knee,
                    right_ankle

                ]


                angle_name = "Knee"


            # ------------------------------------------------
            # LIMIT SCORE
            # ------------------------------------------------

            score = max(
                0,
                min(100, score)
            )


            # ------------------------------------------------
            # DRAW POSE POINTS
            # ------------------------------------------------

            for point in points:

                cv2.circle(
                    image,
                    point,
                    6,
                    (0, 255, 0),
                    -1
                )


            # ------------------------------------------------
            # CAMERA TEXT
            # ------------------------------------------------

            cv2.putText(
                image,
                exercise,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


            with state.lock:

                reps = state.reps


            cv2.putText(
                image,
                f"REPS: {reps}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )


            cv2.putText(
                image,
                f"{angle_name}: {int(angle)}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            cv2.putText(
                image,
                f"FORM: {score}/100",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


            # ------------------------------------------------
            # SAVE STATE
            # ------------------------------------------------

            with state.lock:

                state.angle = angle

                state.score = score

                state.feedback = feedback


        # ====================================================
        # NO PERSON
        # ====================================================

        else:

            with state.lock:

                state.angle = 0

                state.score = 0

                state.feedback = (
                    "Please show your full body to the camera."
                )


            cv2.putText(
                image,
                "NO PERSON DETECTED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


        return frame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🏋️ AI Home Workout"
    )

    st.markdown(
        "<p style='color:#8e9bad;'>"
        "Smart fitness assistant"
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")


    exercise = st.selectbox(
        "Choose Exercise",
        [
            "Push-up",
            "Squat"
        ]
    )


    with state.lock:

        state.exercise = exercise


    st.markdown("---")

    st.markdown("### Instructions")

    st.write("1️⃣ Select exercise")

    st.write("2️⃣ Start camera")

    st.write("3️⃣ Allow camera permission")

    st.write("4️⃣ Show your full body")

    st.write("5️⃣ Start workout")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'AI Home Workout'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your personal AI-powered fitness assistant'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero-card">

        <div class="hero-title">
            🔥 {exercise} Session
        </div>

        <div class="hero-text">
            Real-time AI pose detection,
            repetition counting and form analysis.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WORKOUT BUTTONS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    if st.button(
        "▶ START WORKOUT",
        use_container_width=True
    ):

        with state.lock:

            state.running = True

            state.reps = 0

            state.stage = "UP"

            state.start_time = time.time()

            state.last_timestamp = 0


with col2:

    if st.button(
        "⏹ STOP WORKOUT",
        use_container_width=True
    ):

        with state.lock:

            state.running = False


# ============================================================
# READ STATE
# ============================================================

with state.lock:

    reps = state.reps

    angle = state.angle

    score = state.score

    feedback = state.feedback

    running = state.running

    start_time = state.start_time


# ============================================================
# TIMER
# ============================================================

if start_time and running:

    elapsed = int(
        time.time() - start_time
    )

else:

    elapsed = 0


minutes = elapsed // 60

seconds = elapsed % 60

timer = f"{minutes:02d}:{seconds:02d}"


# ============================================================
# CALORIES
# ============================================================

calories = calorie_estimate(
    reps,
    exercise
)


# ============================================================
# DASHBOARD
# ============================================================

st.markdown(
    "### 📊 Live Workout"
)


m1, m2, m3, m4 = st.columns(4)


with m1:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                🔢 Repetitions
            </div>

            <div class="metric-value">
                {reps}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m2:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                ⏱️ Workout Time
            </div>

            <div class="metric-value">
                {timer}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m3:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                🔥 Calories
            </div>

            <div class="metric-value">
                {calories:.1f}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m4:

    score_text = (
        f"{score}/100"
        if score > 0
        else "--"
    )

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                ⭐ Form Score
            </div>

            <div class="metric-value">
                {score_text}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# AI FEEDBACK
# ============================================================

st.markdown(
    f"""
    <div class="feedback-card">

        🤖 <b>AI Coach:</b>
        {feedback}

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WEBRTC CAMERA
# ============================================================

st.markdown(
    "### 📷 Live AI Camera"
)


st.info(
    "Press START below and allow camera permission."
)


RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


webrtc_ctx = webrtc_streamer(

    key="workout-camera",

    mode=WebRtcMode.SENDRECV,

    rtc_configuration=RTC_CONFIGURATION,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    video_processor_factory=WorkoutProcessor,

    async_processing=True
)


# ============================================================
# SUMMARY
# ============================================================

st.markdown("---")

st.markdown(
    "### 📊 Workout Summary"
)


s1, s2, s3 = st.columns(3)


with s1:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Exercise
            </div>

            <div class="metric-value"
                 style="font-size:25px;">

                {exercise}

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with s2:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Total Reps
            </div>

            <div class="metric-value">

                {reps}

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with s3:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Estimated Calories
            </div>

            <div class="metric-value">

                {calories:.1f}

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        padding:15px;
    ">

        🏋️ AI Home Workout
        &nbsp; • &nbsp;
        MediaPipe AI
        &nbsp; • &nbsp;
        Real-Time Pose Detection

    </div>
    """,
    unsafe_allow_html=True
)