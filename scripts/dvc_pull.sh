#!/bin/bash
# ============================================================================
# DVC PULL AUTOMATION SCRIPT
# ============================================================================
# Este script automatiza el proceso de pull de datos, features y modelos
# desde remote storage DVC. Incluye validaciones, logging y manejo de errores.
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# VALIDACIONES PREVIAS
# ============================================================================

log_info "Iniciando proceso de DVC Pull..."

# Verificar que estamos en el directorio raíz del proyecto
if [ ! -f "dvc.yaml" ]; then
    log_error "No se encontró dvc.yaml. Ejecuta este script desde la raíz del proyecto."
    exit 1
fi

# Verificar que DVC está instalado
if ! command -v dvc &> /dev/null; then
    log_error "DVC no está instalado. Instala con: pip install dvc"
    exit 1
fi

# Verificar que hay un remote configurado
REMOTE=$(dvc remote list | head -1)
if [ -z "$REMOTE" ]; then
    log_error "No hay remote configurado en DVC. Configura uno con: dvc remote add"
    exit 1
fi

log_success "Validaciones previas completadas"

# ============================================================================
# VERIFICAR STATUS DE DVC
# ============================================================================

log_info "Verificando status de DVC..."
dvc status || log_warning "Algunos archivos pueden estar desactualizados"

# ============================================================================
# PULL DE DATOS Y MODELOS
# ============================================================================

log_info "Ejecutando DVC pull desde: $REMOTE"

# Opción 1: Pull completo (todos los archivos trackeados)
if [ "$1" == "--all" ] || [ -z "$1" ]; then
    log_info "Modo: Pull completo de todos los artefactos"
    dvc pull

# Opción 2: Pull solo datos raw
elif [ "$1" == "--data-raw" ]; then
    log_info "Modo: Pull solo datos raw"
    dvc pull data/01_raw.dvc

# Opción 3: Pull solo features procesadas
elif [ "$1" == "--features" ]; then
    log_info "Modo: Pull solo features"
    dvc pull data/04_feature.dvc

# Opción 4: Pull solo modelos
elif [ "$1" == "--models" ]; then
    log_info "Modo: Pull solo modelos"
    dvc pull data/06_models.dvc

# Opción 5: Pull datos completos (sin modelos)
elif [ "$1" == "--data" ]; then
    log_info "Modo: Pull datos completos (sin modelos)"
    dvc pull data/01_raw.dvc data/02_intermediate.dvc data/03_primary.dvc data/04_feature.dvc

# Opción 6: Pull específico
else
    log_info "Modo: Pull específico de $1"
    dvc pull "$1"
fi

if [ $? -eq 0 ]; then
    log_success "DVC pull completado exitosamente"
else
    log_error "DVC pull falló"
    exit 1
fi

# ============================================================================
# VERIFICAR INTEGRIDAD DE DATOS
# ============================================================================

log_info "Verificando integridad de datos descargados..."

# Contar archivos en data/
if [ -d "data/" ]; then
    RAW_COUNT=$(find data/01_raw -type f 2>/dev/null | wc -l)
    FEATURE_COUNT=$(find data/04_feature -type f 2>/dev/null | wc -l)
    MODEL_COUNT=$(find data/06_models -type f 2>/dev/null | wc -l)

    log_info "Archivos descargados:"
    log_info "  - Raw data: $RAW_COUNT archivos"
    log_info "  - Features: $FEATURE_COUNT archivos"
    log_info "  - Modelos: $MODEL_COUNT archivos"
else
    log_warning "Directorio data/ no encontrado"
fi

# ============================================================================
# RESUMEN Y PRÓXIMOS PASOS
# ============================================================================

echo ""
log_success "=========================================="
log_success "DVC Pull completado exitosamente"
log_success "=========================================="
log_info "Remote usado: $REMOTE"
log_info "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
log_info "Próximos pasos:"
log_info "  1. Verifica que los datos se descargaron correctamente"
log_info "  2. Ejecuta los pipelines de Kedro si es necesario:"
log_info "     - kedro run --pipeline=regresion"
log_info "     - kedro run --pipeline=classification"
log_info "  3. O visualiza con: kedro viz"
echo ""

# ============================================================================
# MODO AVANZADO: Verificación de checksums
# ============================================================================

if [ "$2" == "--verify" ]; then
    log_info "Verificando checksums MD5..."
    dvc status
    if [ $? -eq 0 ]; then
        log_success "Todos los checksums son válidos"
    else
        log_warning "Algunos archivos tienen checksums diferentes"
    fi
fi

# ============================================================================
# USO
# ============================================================================
# ./scripts/dvc_pull.sh                    # Pull completo
# ./scripts/dvc_pull.sh --all              # Pull completo (explícito)
# ./scripts/dvc_pull.sh --data-raw         # Pull solo datos raw
# ./scripts/dvc_pull.sh --features         # Pull solo features
# ./scripts/dvc_pull.sh --models           # Pull solo modelos
# ./scripts/dvc_pull.sh --data             # Pull datos completos (sin modelos)
# ./scripts/dvc_pull.sh data/06_models.dvc # Pull específico
# ./scripts/dvc_pull.sh --all --verify     # Pull con verificación de checksums
# ============================================================================
