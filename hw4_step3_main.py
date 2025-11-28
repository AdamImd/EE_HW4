"""
EE HW4 Step 3: Stable Diffusion Implementation
This script implements text-to-image generation using Stable Diffusion with:
- Configurable CFG scale, sampling steps, and eta
- CLIP-based similarity scoring
- Support for positive and negative prompts
"""

import torch
import argparse
import os
import numpy as np
from PIL import Image
import json
from datetime import datetime

# Hugging Face imports
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Results directory
RESULTS_DIR = "results/step3_results"


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # For complete reproducibility with CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_stable_diffusion_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5"):
    """
    Load Stable Diffusion pipeline with DDIM scheduler for eta control.
    
    Args:
        model_id: Hugging Face model identifier
        
    Returns:
        StableDiffusionPipeline with DDIM scheduler
    """
    print(f"Loading Stable Diffusion model: {model_id}")
    
    # Load the pipeline with DDIM scheduler to allow eta control
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,  # Disable safety checker for research
        requires_safety_checker=False
    )
    
    # Replace scheduler with DDIM for eta control
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    pipe = pipe.to(DEVICE)
    
    # Enable memory efficient attention if available
    if DEVICE == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
        except:
            print("xformers not available, using default attention")
    
    return pipe


def load_clip_model(model_id: str = "openai/clip-vit-base-patch32"):
    """
    Load CLIP model for similarity scoring.
    
    Args:
        model_id: Hugging Face CLIP model identifier
        
    Returns:
        Tuple of (CLIPModel, CLIPProcessor)
    """
    print(f"Loading CLIP model: {model_id}")
    model = CLIPModel.from_pretrained(model_id).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


def compute_clip_similarity(image: Image.Image, text: str, clip_model, clip_processor) -> float:
    """
    Compute CLIP similarity score between an image and text prompt.
    
    Args:
        image: PIL Image
        text: Text prompt
        clip_model: Loaded CLIP model
        clip_processor: CLIP processor
        
    Returns:
        Cosine similarity score (0-1 range, multiplied by 100 for percentage)
    """
    # Process inputs with truncation to handle long prompts (CLIP max is 77 tokens)
    inputs = clip_processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(DEVICE)
    
    # Get embeddings
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = torch.matmul(image_embeds, text_embeds.T).item()
    
    return similarity * 100  # Convert to percentage scale


def generate_image(
    pipe,
    prompt: str = "",
    negative_prompt: str = None,
    cfg_scale: float = 7.5,
    num_steps: int = 50,
    eta: float = 0.0,
    seed: int = 42,
    height: int = 512,
    width: int = 512
) -> Image.Image:
    """
    Generate an image using Stable Diffusion.
    
    Args:
        pipe: StableDiffusionPipeline
        prompt: Positive text prompt
        negative_prompt: Negative text prompt (optional)
        cfg_scale: Classifier-free guidance scale (omega_CFG)
        num_steps: Number of sampling steps
        eta: DDIM eta parameter (0=deterministic, 1=stochastic)
        seed: Random seed for reproducibility
        height: Output image height
        width: Output image width
        
    Returns:
        Generated PIL Image
    """
    set_seed(seed)
    
    # Create generator for reproducibility
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    # Handle empty prompt for unconditional generation
    if not prompt or prompt.strip() == "":
        prompt = ""
    
    # Generate image
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=cfg_scale,
        num_inference_steps=num_steps,
        eta=eta,
        generator=generator,
        height=height,
        width=width
    )
    
    return result.images[0]


def save_image_with_metadata(
    image: Image.Image,
    save_path: str,
    prompt: str,
    negative_prompt: str,
    cfg_scale: float,
    num_steps: int,
    eta: float,
    seed: int,
    clip_score: float = None,
    manual_score: float = None
):
    """Save image and its generation metadata."""
    # Save image
    image.save(save_path)
    
    # Save metadata
    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "cfg_scale": cfg_scale,
        "num_steps": num_steps,
        "eta": eta,
        "seed": seed,
        "clip_score": clip_score,
        "manual_score": manual_score,
        "timestamp": datetime.now().isoformat()
    }
    
    metadata_path = save_path.replace(".png", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


# ============================================================================
# Part (a): Unconditional generation test
# ============================================================================

def run_part_a(pipe, save_dir: str, seed: int = 42):
    """
    Part (a): Test unconditional generation with CFG=0.0, eta=0.0, 50 steps.
    """
    print("\n" + "="*60)
    print("Part (a): Unconditional Generation Test")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate with unconditional settings
    image = generate_image(
        pipe,
        prompt="",
        negative_prompt=None,
        cfg_scale=0.0,  # No classifier-free guidance
        num_steps=50,
        eta=0.0,
        seed=seed
    )
    
    save_path = os.path.join(save_dir, "part_a_unconditional.png")
    save_image_with_metadata(
        image, save_path,
        prompt="",
        negative_prompt=None,
        cfg_scale=0.0,
        num_steps=50,
        eta=0.0,
        seed=seed
    )
    
    print(f"Unconditional image saved to: {save_path}")
    
    # Verify reproducibility by generating again
    image_verify = generate_image(
        pipe,
        prompt="",
        negative_prompt=None,
        cfg_scale=0.0,
        num_steps=50,
        eta=0.0,
        seed=seed
    )
    
    # Check if images are identical
    img_array1 = np.array(image)
    img_array2 = np.array(image_verify)
    is_identical = np.allclose(img_array1, img_array2)
    print(f"Reproducibility check (same seed produces identical output): {is_identical}")
    
    return image


# ============================================================================
# Part (b): 15 Prompts across 5 Topics
# ============================================================================

def get_prompts():
    """
    Define 15 prompts across 5 topics (3 versions each: simple, medium, detailed).
    These prompts are manually created for creativity and diversity.
    """
    prompts = {
        # Topic 1: Space Exploration
        "space_exploration": {
            "simple": "An astronaut  in space",
            "medium": "An astronaut in a spacesuit floating above Earth",
            "detailed": "A single astronaut in a shiny white spacesuit drifting serenely against the stars in the sky. There is a silent planet below with swirling clouds and blue oceans, with their ship orbiting in the distance"
        },
        # Topic 2: Underwater World
        "underwater_world": {
            "simple": "A coral reef",
            "medium": "A colorful coral reef with tropical fish of and sunlight filtering through the water",
            "detailed": "An underwater coral reef that has all sorts of life, with many fish and sharks swimming around. It has bright corals of all colors and shapes, with sunlight filtering through the clear blue water from above."
        },
        # Topic 3: Ancient Architecture
        "ancient_architecture": {
            "simple": "A medieval castle",
            "medium": "A lively medieval castle surrounded by a moat and lush greenery",
            "detailed": "A beautyful german day, with a large castle made of stone, with a few vines climbing up the spires. The vilage around the castle is full of life, with people walking around the market."
        },
        # Topic 4: Futuristic City
        "futuristic_city": {
            "simple": "A cyberpunk city",
            "medium": "A distopian cyberpunk city, with neon lights and flying cars.",
            "detailed": "A breathtaking cyberpunk megacity that has bustling streets filled with people and vendors. The skyline has many towering skyscrapers, and there are futuristic flying cars."
        },
        # Topic 5: Fantasy Creatures
        "fantasy_creatures": {
            "simple": "A cat with a bird body",
            "medium": "A chimera with the body of a cat, wings of a bird.",
            "detailed": "A beautiful chimera creature that has the body of a maine coon cat, with large majestic wings of an eagle."
        }
    }
    
    return prompts


def run_part_b(pipe, clip_model, clip_processor, save_dir: str, seed: int = 42):
    """
    Part (b): Generate images for 15 prompts and compute CLIP scores.
    Returns results for comparison with manual scores.
    """
    print("\n" + "="*60)
    print("Part (b): 15 Prompts Generation (5 Topics × 3 Versions)")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    prompts = get_prompts()
    results = []
    
    # Manual similarity scores (to be filled by user after reviewing images)
    # These are placeholder values - UPDATE THESE after reviewing generated images
    manual_scores = {
        "space_exploration": {"simple": 7.5, "medium": 8.0, "detailed": 7.0},
        "underwater_world": {"simple": 8.0, "medium": 8.5, "detailed": 7.5},
        "ancient_architecture": {"simple": 8.5, "medium": 9.0, "detailed": 8.0},
        "futuristic_city": {"simple": 7.0, "medium": 8.0, "detailed": 7.5},
        "fantasy_creatures": {"simple": 8.0, "medium": 8.5, "detailed": 7.5}
    }
    
    for topic_idx, (topic, versions) in enumerate(prompts.items(), 1):
        print(f"\n--- Topic {topic_idx}/5: {topic.replace('_', ' ').title()} ---")
        
        for version_name, prompt in versions.items():
            print(f"\n  Generating: {version_name} version")
            print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            
            # Generate image
            image = generate_image(
                pipe,
                prompt=prompt,
                negative_prompt=None,
                cfg_scale=10.0,
                num_steps=50,
                eta=0.0,
                seed=seed
            )
            
            # Compute CLIP score
            clip_score = compute_clip_similarity(image, prompt, clip_model, clip_processor)
            
            # Get manual score
            manual_score = manual_scores[topic][version_name]
            
            # Save image
            filename = f"part_b_{topic}_{version_name}.png"
            save_path = os.path.join(save_dir, filename)
            save_image_with_metadata(
                image, save_path,
                prompt=prompt,
                negative_prompt=None,
                cfg_scale=10.0,
                num_steps=50,
                eta=0.0,
                seed=seed,
                clip_score=clip_score,
                manual_score=manual_score
            )
            
            result = {
                "topic": topic,
                "version": version_name,
                "prompt": prompt,
                "clip_score": clip_score,
                "manual_score": manual_score,
                "image_path": save_path
            }
            results.append(result)
            
            print(f"  CLIP Score: {clip_score:.2f}%")
            print(f"  Manual Score: {manual_score}/10")
            print(f"  Saved: {filename}")
    
    # Save all results to JSON
    results_json_path = os.path.join(save_dir, "part_b_results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# ============================================================================
# Part (c): CLIP Similarity Analysis
# ============================================================================

def run_part_c(results: list, save_dir: str):
    """
    Part (c): Analyze CLIP scores vs manual scores.
    """
    print("\n" + "="*60)
    print("Part (c): CLIP Similarity Analysis")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Analyze by version type
    version_analysis = {"simple": [], "medium": [], "detailed": []}
    for r in results:
        version_analysis[r["version"]].append({
            "clip": r["clip_score"],
            "manual": r["manual_score"]
        })
    
    print("\n--- Analysis by Prompt Length ---")
    for version, scores in version_analysis.items():
        avg_clip = np.mean([s["clip"] for s in scores])
        avg_manual = np.mean([s["manual"] for s in scores])
        print(f"\n{version.capitalize()} prompts:")
        print(f"  Average CLIP Score: {avg_clip:.2f}%")
        print(f"  Average Manual Score: {avg_manual:.1f}/10")
    
    # Correlation analysis
    clip_scores = [r["clip_score"] for r in results]
    manual_scores = [r["manual_score"] * 10 for r in results]  # Scale to 0-100
    
    correlation = np.corrcoef(clip_scores, manual_scores)[0, 1]
    print(f"\n--- Correlation Analysis ---")
    print(f"Pearson correlation between CLIP and Manual scores: {correlation:.3f}")
    
    # Save analysis
    analysis = {
        "version_averages": {
            version: {
                "avg_clip": float(np.mean([s["clip"] for s in scores])),
                "avg_manual": float(np.mean([s["manual"] for s in scores]))
            }
            for version, scores in version_analysis.items()
        },
        "correlation": float(correlation),
        "all_scores": [
            {"prompt": r["prompt"][:50], "clip": r["clip_score"], "manual": r["manual_score"]}
            for r in results
        ]
    }
    
    analysis_path = os.path.join(save_dir, "part_c_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nAnalysis saved to: {analysis_path}")
    
    return analysis


# ============================================================================
# Part (d): Negative Prompts and CFG Scale Comparison
# ============================================================================

def run_part_d(pipe, clip_model, clip_processor, save_dir: str, seed: int = 42):
    """
    Part (d): Test different CFG scales with a negative prompt.
    """
    print("\n" + "="*60)
    print("Part (d): Negative Prompt and CFG Scale Analysis")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Select one prompt (using the detailed space exploration prompt)
    prompts = get_prompts()
    selected_prompt = prompts["futuristic_city"]["medium"]
    
    # Meaningful negative prompt to avoid common artifacts and unwanted elements
    negative_prompt = "blurry, low quality, distorted, ugly, deformed, bad anatomy, low resolution, grainy, noisy, watermark, text, signature, poorly drawn, amateur, disfigured"
    
    cfg_scales = [0.0, 2.0, 5.0, 8.0, 12.0, 15.0]
    results = []
    
    print(f"\nSelected Prompt: {selected_prompt}")
    print(f"Negative Prompt: {negative_prompt}")
    
    for cfg_scale in cfg_scales:
        print(f"\n--- Generating with CFG Scale: {cfg_scale} ---")
        
        image = generate_image(
            pipe,
            prompt=selected_prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            num_steps=50,
            eta=0.0,
            seed=seed
        )
        
        # Compute CLIP score
        clip_score = compute_clip_similarity(image, selected_prompt, clip_model, clip_processor)
        
        # Save image
        filename = f"part_d_cfg_{cfg_scale:.1f}.png"
        save_path = os.path.join(save_dir, filename)
        save_image_with_metadata(
            image, save_path,
            prompt=selected_prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            num_steps=50,
            eta=0.0,
            seed=seed,
            clip_score=clip_score
        )
        
        result = {
            "cfg_scale": cfg_scale,
            "clip_score": clip_score,
            "image_path": save_path
        }
        results.append(result)
        
        print(f"  CLIP Score: {clip_score:.2f}%")
        print(f"  Saved: {filename}")
    
    # Save results
    results_data = {
        "prompt": selected_prompt,
        "negative_prompt": negative_prompt,
        "num_steps": 50,
        "eta": 0.0,
        "seed": seed,
        "results": results
    }
    
    results_path = os.path.join(save_dir, "part_d_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    
    # Print summary
    print("\n--- CFG Scale vs CLIP Score Summary ---")
    for r in results:
        print(f"  CFG {r['cfg_scale']:5.1f}: CLIP Score = {r['clip_score']:.2f}%")
    
    return results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EE HW4 Step 3: Stable Diffusion")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Stable Diffusion model ID from Hugging Face")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP model ID for similarity scoring")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--parts", type=str, default="abcd",
                        help="Which parts to run (e.g., 'ab' for parts a and b)")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR,
                        help="Directory to save results")
    args = parser.parse_args()
    
    print("="*60)
    print("EE HW4 Step 3: Stable Diffusion Implementation")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"SD Model: {args.model}")
    print(f"CLIP Model: {args.clip_model}")
    print(f"Seed: {args.seed}")
    print(f"Running parts: {args.parts}")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load models
    pipe = load_stable_diffusion_pipeline(args.model)
    
    # Load CLIP model (needed for parts b, c, d)
    clip_model, clip_processor = None, None
    if any(p in args.parts.lower() for p in ['b', 'c', 'd']):
        clip_model, clip_processor = load_clip_model(args.clip_model)
    
    # Run requested parts
    results_b = None
    
    if 'a' in args.parts.lower():
        part_a_dir = os.path.join(args.results_dir, "part_a")
        run_part_a(pipe, part_a_dir, args.seed)
    
    if 'b' in args.parts.lower():
        part_b_dir = os.path.join(args.results_dir, "part_b")
        results_b = run_part_b(pipe, clip_model, clip_processor, part_b_dir, args.seed)
    
    if 'c' in args.parts.lower():
        part_c_dir = os.path.join(args.results_dir, "part_c")
        if results_b is None:
            # Load results from part b if not run in this session
            part_b_results_path = os.path.join(args.results_dir, "part_b", "part_b_results.json")
            if os.path.exists(part_b_results_path):
                with open(part_b_results_path, "r") as f:
                    results_b = json.load(f)
            else:
                print("Warning: Part B results not found. Run part B first.")
                results_b = []
        run_part_c(results_b, part_c_dir)
    
    if 'd' in args.parts.lower():
        part_d_dir = os.path.join(args.results_dir, "part_d")
        run_part_d(pipe, clip_model, clip_processor, part_d_dir, args.seed)
    
    print("\n" + "="*60)
    print("All requested parts completed!")
    print(f"Results saved to: {args.results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
