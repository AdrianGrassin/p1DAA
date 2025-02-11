# Multiplicación de Matrices

Este proyecto implementa diferentes estrategias de multiplicación de matrices utilizando el patrón de diseño Strategy, incluyendo implementaciones en CPU y GPU.

## Características

- Multiplicación de matrices por filas (CPU)
- Multiplicación de matrices por columnas (CPU)
- Multiplicación de matrices en GPU (AMD/NVIDIA)
- Multiplicación híbrida CPU-GPU
- Generación automática de matrices de prueba
- Benchmarking detallado con métricas de rendimiento
- Soporte para múltiples tamaños de matriz

## Requisitos

- .NET 8.0 SDK
- GPU compatible (AMD o NVIDIA)
- Python 3.x (para generación de gráficas)
- Paquetes Python: pandas, matplotlib

## Instalación

### Windows
```powershell
# Instalación de controladores GPU
./scripts/install_windows.ps1

# Verificar instalación
./scripts/check-opencl.ps1
./scripts/detect-gpu.ps1
```

### Linux
```bash
# Instalación de controladores GPU
./scripts/install_linux.sh

# Verificar instalación
./scripts/detect-gpu.sh
```

## Uso

1. Ejecutar los benchmarks:
```bash
dotnet run
```

2. Generar gráficas de rendimiento:
```bash
python scripts/generate_graph.py
```

## Resultados de Rendimiento

- Matrices pequeñas (<500x500): Mejor rendimiento con CPU por filas
- Matrices medianas (500x500-1000x1000): GPU muestra ventajas significativas
- Matrices grandes (>1000x1000): GPU y modo híbrido dominan el rendimiento

## Estructura del Proyecto

- `Core/`: Interfaces y clases base
- `GPU/`: Implementaciones específicas para AMD y NVIDIA
- `docs/`: Documentación y gráficas
- `scripts/`: Scripts de utilidad y generación de gráficas
- `Install/`: Scripts de instalación

## Contribuir

Las contribuciones son bienvenidas. Por favor, asegúrate de actualizar las pruebas según corresponda.