# Building and Using the Base Image

This addon uses a pre-built base image with PyTorch and NeuralProphet to speed up builds.

## Option 1: Automated (GitHub Actions)

The base image is automatically built and pushed to GitHub Container Registry when:
- You push changes to `Dockerfile.base` or `requirements.txt`
- You manually trigger the workflow from Actions tab

The workflow builds for both amd64 and aarch64 architectures.

## Option 2: Manual Build

### Build locally for testing:

```bash
# For amd64
cd neuralprophet
docker buildx build --platform linux/amd64 \
  -f Dockerfile.base \
  -t ghcr.io/sunnysideal/neuralprophet-base:amd64 \
  --build-arg BUILD_ARCH=amd64 \
  .

# For aarch64 (ARM64)
docker buildx build --platform linux/arm64 \
  -f Dockerfile.base \
  -t ghcr.io/sunnysideal/neuralprophet-base:aarch64 \
  --build-arg BUILD_ARCH=aarch64 \
  .
```

### Push to registry:

```bash
# Login first
docker login ghcr.io -u sunnysideal

# Push both architectures
docker push ghcr.io/sunnysideal/neuralprophet-base:amd64
docker push ghcr.io/sunnysideal/neuralprophet-base:aarch64
```

## Using the Base Image

### In Dockerfile:

Uncomment these lines in `Dockerfile`:
```dockerfile
ARG BUILD_ARCH=amd64
FROM ghcr.io/sunnysideal/neuralprophet-base:${BUILD_ARCH}
```

And comment out:
```dockerfile
# ARG BUILD_FROM
# FROM $BUILD_FROM
```

### In config.yaml:

Update the image reference:
```yaml
image: "ghcr.io/sunnysideal/neuralprophet-base/{arch}"
```

## Benefits

- **Fast builds**: ~30 seconds instead of 5-10 minutes
- **Reliable**: No dependency resolution issues during addon installation
- **Consistent**: Same dependencies across all installations
- **Offline-capable**: Dependencies pre-downloaded

## Updating Dependencies

1. Update `requirements.txt`
2. Commit and push (triggers automatic rebuild)
3. Wait for GitHub Actions to complete (~10 minutes)
4. New addon builds will use updated base image
