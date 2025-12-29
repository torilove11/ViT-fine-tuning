import gradio as gr
import google.generativeai as genai
import os
from dotenv import load_dotenv
from infer import predict

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(".env 파일에 GEMINI_API_KEY가 없습니다.")

genai.configure(api_key=GEMINI_API_KEY)

llm_model = genai.GenerativeModel('gemini-flash-latest')

def get_dog_physiognomy(image, breed_name):
    """
    이미지와 모델의 예측 정보를 받아 Gemini API를 이용해 강아지의 관상을 봅니다.
    """
    print(f"Gemini에게 관상 요청 중... 견종: {breed_name}")

    prompt = f"""
    ### [Role & Instruction]
    당신은 신통방통하기로 소문난 'AI 강아지 관상가'입니다. 
    입력된 강아지 견종인 [{breed_name}]의 외형적 특징, 역사적 배경, 성격을 관상학적 관점에서 재해석하여 흥미롭고 따뜻한 분석을 제공해야 합니다.
    함께 제공된 강아지 사진의 표정과 분위기도 반영해주세요.

    지침:
    1. 말투는 전문 관상가처럼 신뢰감 있으면서도 다정하게(예: ~구나, ~할 상이로다) 작성하세요.
    2. 해당 견종의 신체적 특징(눈매, 귀 모양, 털색 등)을 관상 용어와 연결하세요.
    3. 이 강아지가 주인에게 어떤 복(재물, 건강, 화목 등)을 가져다줄지 마무리 멘트를 작성하세요.
    4. 분량은 200자 내외로 작성하세요.

    ### [Few-shot Examples]

    Input: 말티즈
    Output: 오호, 눈망울이 밤다래처럼 맑고 초롱초롱한 것을 보니 영락없는 '청명상(淸明象)'이로구나! 하얀 털은 집안의 나쁜 기운을 정화하는 기운을 가졌으니, 이 아이가 머무는 곳마다 웃음꽃이 피어날 것이야. 주인에게는 맑은 정신과 평안을 가져다줄 복덩이이니 애지중지 아끼도록 하거라.

    Input: 시바견
    Output: 다부진 입매와 쫑긋하게 선 귀가 재물을 꽉 잡고 놓지 않을 '수재상(守財象)'이로다! 고집이 있어 보이나 이는 주인에 대한 일편단심 충성심이 깊다는 증거지. 힘차게 말려 올라간 저 꼬리는 집안의 복이 밖으로 새나가지 않게 꽉 붙들어 매는 역할을 하니, 너의 재물운을 책임질 든든한 파트너가 될 상이야.

    Input: 골든 리트리버
    Output: 허허, 얼굴 가득 온화함이 넘쳐흐르니 주위 사람들을 모두 행복하게 만드는 '화합상(和合象)'이로다. 축 처진 귀는 타인의 슬픔을 잘 들어주는 깊은 배려심을 뜻하고, 황금빛 털은 집안에 황금 같은 풍요를 가져올 징조니라. 이 아이와 함께라면 만사가 형통하고 가정이 화목해질 것이니 최고의 귀인을 만난 셈이야.

    ### [Current Input]
    Input: {breed_name}
    Output:
    """

    try:
        # 이미지와 텍스트 프롬프트를 함께 전달합니다.
        response = llm_model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"API 오류로 관상을 볼 수 없습니다.\n(에러: {str(e)})"


def process_step(image):
    if image is None:
        return None, "강아지 사진을 올려주세요."
    
    predictions = predict(image) 
    
    # Gemini에게는 1순위 견종만 전송합니다.
    top_breed = list(predictions.keys())[0]
    fortune_telling = get_dog_physiognomy(image, top_breed)
    
    return predictions, fortune_telling


custom_css = """
.container { max-width: 900px; margin: auto; }
.output-markdown { background-color: #fdfbf7; border: 1px solid #e0d8c8; padding: 20px; border-radius: 10px; font-family: 'KoPub Batang', serif; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    
    gr.Markdown("# 🐕 AI 강아지 관상소")
    gr.Markdown("Fine-tuning된 ViT 모델이 견종을 맞추고, Gemini가 관상을 봐드립니다.")
    
    with gr.Row():
        # [왼쪽] 이미지 입력
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="강아지 사진 업로드")
            submit_btn = gr.Button("🔮 관상 보기", variant="primary", size="lg")
            
        # [오른쪽] 결과 출력
        with gr.Column(scale=1):
            # 견종 결과
            lbl_output = gr.Label(num_top_classes=3, label="견종 분석 결과")
            # 관상 결과
            txt_output = gr.Markdown(label="📢 관상 풀이")

    submit_btn.click(
        fn=process_step,
        inputs=input_image,
        outputs=[lbl_output, txt_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)