import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import time

# Device configuration
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_device()

# Set up Streamlit page
st.set_page_config(page_title="Hybrid AI Art Generator", page_icon="🎨")
st.title("🎨 Hybrid CPU/GPU Art Generator")
st.write(f"Running on: **{device.upper()}**")

# Model loading with hybrid support
@st.cache_resource
def load_model(model_name="runwayml/stable-diffusion-v1-5"):
    try:
        model_args = {
            "pretrained_model_name_or_path": model_name,
            "safety_checker": None  # Disable to save memory
        }

        if device == "cuda":
            model_args.update({
                "torch_dtype": torch.float16,
                "variant": "fp16",
            })
        
        pipe = StableDiffusionPipeline.from_pretrained(**model_args)
        
        # Enable CPU offloading if using GPU
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
            
        # Optimizations
        pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        
        return pipe
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Load model
with st.spinner(f"Loading model for {device.upper()}..."):
    pipe = load_model()
    if pipe is None:
        st.error("Failed to load model. Please check your internet connection and try again.")
        st.stop()

# Hybrid generation function
def generate_images(prompt, negative_prompt, steps, guidance_scale, num_images):
    results = []
    
    for i in range(num_images):
        try:
            start_time = time.time()
            
            # Generate image
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            gen_time = time.time() - start_time
            st.sidebar.write(f"Image {i+1} generated in {gen_time:.1f}s")
            results.append(image)
            
            # Clear memory between generations
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            st.error(f"Error generating image {i+1}: {str(e)}")
    
    return results

# UI Components
with st.sidebar:
    st.header("Settings")
    num_images = st.slider("Number of images", 1, 4, 1)
    steps = st.slider("Steps", 10, 100, 50)
    guidance_scale = st.slider("Creativity", 1.0, 20.0, 7.5)
    seed = st.number_input("Random seed (0 for random)", 0)
    
    st.header("System Info")
    st.write(f"Device: {device.upper()}")
    if device == "cuda":
        st.write(f"GPU: {torch.cuda.get_device_name(0)}")
        st.write(f"Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB used")

prompt = st.text_area(
    "Describe your artwork",
    "A beautiful landscape with mountains and a lake at sunset, digital art",
    height=100
)

negative_prompt = st.text_input(
    "Things to avoid in the image",
    "blurry, low quality, distorted, ugly"
)

if st.button("Generate Art"):
    if not prompt.strip():
        st.warning("Please enter a prompt")
    else:
        with st.spinner(f"Generating {num_images} image(s)..."):
            try:
                # Set seed if specified
                if seed != 0:
                    torch.manual_seed(seed)
                
                # Generate images
                images = generate_images(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    num_images=num_images
                )
                
                # Display results
                cols = st.columns(min(2, num_images))
                for i, img in enumerate(images):
                    with cols[i % len(cols)]:
                        st.image(img, use_container_width=True)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        st.download_button(
                            f"Download Image {i+1}",
                            buf.getvalue(),
                            f"ai_art_{i+1}.png",
                            "image/png"
                        )
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
            finally:
                if device == "cuda":
                    torch.cuda.empty_cache()

# Tips section
with st.expander("Optimization Tips"):
    st.markdown("""
    - **For Windows GPU users**: Enable 'Hardware-accelerated GPU scheduling' in Windows settings
    - **Reduce steps**: 20-30 steps often gives good results
    - **Memory issues**: Try lowering steps/image count first
    - **Hybrid mode**: Automatically uses CPU offloading when GPU memory is full
    """)