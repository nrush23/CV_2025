"""
Test script to verify all OASIS-Pong components
"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def test_vae_encoder():
    """Test VAE Encoder"""
    print("\n" + "=" * 60)
    print("Testing VAE Encoder...")
    print("=" * 60)
    
    from vae_encoder import create_vae_encoder
    
    encoder = create_vae_encoder()
    encoder.eval()
    
    # Test forward pass
    x = torch.randn(2, 3, 210, 160)
    mu, logvar = encoder(x)
    
    assert mu.shape == (2, 21*16, 16), f"Unexpected mu shape: {mu.shape}"
    assert logvar.shape == (2, 21*16, 16), f"Unexpected logvar shape: {logvar.shape}"
    
    # Test reparameterization
    z = encoder.reparameterize(mu, logvar)
    assert z.shape == mu.shape, f"Unexpected z shape: {z.shape}"
    
    # Test encode function
    z, mu, logvar = encoder.encode(x)
    assert z.shape == (2, 21*16, 16), f"Unexpected encoded shape: {z.shape}"
    
    print("✅ VAE Encoder passed all tests!")
    return encoder


def test_vae_decoder():
    """Test VAE Decoder"""
    print("\n" + "=" * 60)
    print("Testing VAE Decoder...")
    print("=" * 60)
    
    from vae_decoder import create_vae_decoder
    
    decoder = create_vae_decoder()
    decoder.eval()
    
    # Test forward pass
    z = torch.randn(2, 21*16, 16)
    recon = decoder(z)
    
    assert recon.shape == (2, 3, 210, 160), f"Unexpected reconstruction shape: {recon.shape}"
    assert recon.min() >= 0 and recon.max() <= 1, "Output not in [0, 1] range"
    
    print("✅ VAE Decoder passed all tests!")
    return decoder


def test_vae():
    """Test complete VAE"""
    print("\n" + "=" * 60)
    print("Testing Complete VAE...")
    print("=" * 60)
    
    from vae_encoder import create_vae_encoder
    from vae_decoder import create_vae_decoder, ViTVAE, vae_loss
    
    encoder = create_vae_encoder()
    decoder = create_vae_decoder()
    vae = ViTVAE(encoder, decoder)
    vae.eval()
    
    # Test forward pass
    x = torch.randn(2, 3, 210, 160)
    recon, mu, logvar = vae(x)
    
    assert recon.shape == x.shape, f"Reconstruction shape mismatch: {recon.shape} vs {x.shape}"
    
    # Test loss
    loss, recon_loss, kld_loss = vae_loss(recon, x, mu, logvar)
    assert loss.item() > 0, "Loss should be positive"
    
    print(f"Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, KLD: {kld_loss.item():.4f})")
    print("✅ Complete VAE passed all tests!")
    return vae


def test_dit():
    """Test DiT model"""
    print("\n" + "=" * 60)
    print("Testing DiT Model...")
    print("=" * 60)
    
    from dit_model import create_dit
    
    dit = create_dit()
    dit.eval()
    
    # Test forward pass
    batch_size = 2
    noisy_latent = torch.randn(batch_size, 21*16, 16)
    timesteps = torch.randint(0, 1000, (batch_size,))
    actions = torch.randint(0, 4, (batch_size,))
    prev_latent = torch.randn(batch_size, 21*16, 16)
    
    with torch.no_grad():
        output = dit(noisy_latent, timesteps, actions, prev_latent)
    
    assert output.shape == noisy_latent.shape, f"Output shape mismatch: {output.shape}"
    
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("✅ DiT Model passed all tests!")
    return dit


def test_ddpm_scheduler():
    """Test DDPM Scheduler"""
    print("\n" + "=" * 60)
    print("Testing DDPM Scheduler...")
    print("=" * 60)
    
    from dit_trainer import DDPMScheduler
    
    scheduler = DDPMScheduler(num_train_timesteps=1000, device='cpu')
    
    # Test add noise
    clean = torch.randn(2, 21*16, 16)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([100, 500])
    
    noisy = scheduler.add_noise(clean, noise, timesteps)
    assert noisy.shape == clean.shape, f"Noisy shape mismatch: {noisy.shape}"
    
    # Test remove noise
    denoised = scheduler.remove_noise(noisy, noise, timesteps)
    assert denoised.shape == clean.shape, f"Denoised shape mismatch: {denoised.shape}"
    
    print("✅ DDPM Scheduler passed all tests!")
    return scheduler


def test_trainers():
    """Test trainers"""
    print("\n" + "=" * 60)
    print("Testing Trainers...")
    print("=" * 60)
    
    from vae_encoder import create_vae_encoder
    from vae_decoder import create_vae_decoder, ViTVAE
    from dit_model import create_dit
    from vae_trainer import VAETrainer, PongFrameDataset
    from dit_trainer import DiTTrainer, PongLatentDataset
    
    # Test VAE Trainer
    encoder = create_vae_encoder()
    decoder = create_vae_decoder()
    vae = ViTVAE(encoder, decoder)
    vae_trainer = VAETrainer(vae, device='cpu')
    
    # Create dummy dataset
    dummy_frames = [np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8) for _ in range(5)]
    dataset = PongFrameDataset(dummy_frames)
    
    # Test single train step
    batch = torch.randn(2, 3, 210, 160)
    loss, recon_loss, kld_loss = vae_trainer.train_step(batch)
    
    print(f"VAE train step - Loss: {loss.item():.4f}")
    print("✅ VAE Trainer works!")
    
    # Test DiT Trainer
    dit = create_dit()
    dit_trainer = DiTTrainer(encoder, dit, device='cpu')
    
    # Test with dummy data
    batch = {
        'latent_t': torch.randn(2, 21*16, 16),
        'action': torch.randint(0, 4, (2,)),
        'latent_next': torch.randn(2, 21*16, 16)
    }
    
    loss = dit_trainer.train_step(batch)
    print(f"DiT train step - Loss: {loss.item():.4f}")
    print("✅ DiT Trainer works!")


def test_pipeline():
    """Test complete pipeline"""
    print("\n" + "=" * 60)
    print("Testing Complete Pipeline...")
    print("=" * 60)
    
    from oasis_pipeline import create_oasis_pipeline
    
    pipeline = create_oasis_pipeline(device='cpu')
    
    print(f"VAE parameters: {sum(p.numel() for p in pipeline.vae.parameters()):,}")
    print(f"DiT parameters: {sum(p.numel() for p in pipeline.dit.parameters()):,}")
    print("✅ Pipeline created successfully!")
    return pipeline


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Running OASIS-Pong Component Tests")
    print("=" * 60)
    
    try:
        # Test individual components
        encoder = test_vae_encoder()
        decoder = test_vae_decoder()
        vae = test_vae()
        dit = test_dit()
        scheduler = test_ddpm_scheduler()
        test_trainers()
        pipeline = test_pipeline()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed!")
        print("=" * 60)
        
        print("\n📊 Model Summary:")
        print(f"VAE Encoder: {sum(p.numel() for p in encoder.parameters()):,} parameters")
        print(f"VAE Decoder: {sum(p.numel() for p in decoder.parameters()):,} parameters")
        print(f"DiT: {sum(p.numel() for p in dit.parameters()):,} parameters")
        print(f"Total: {sum(p.numel() for p in vae.parameters()) + sum(p.numel() for p in dit.parameters()):,} parameters")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
