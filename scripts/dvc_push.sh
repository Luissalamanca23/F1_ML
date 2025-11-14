#!/bin/bash
# ============================================================================
# DVC PUSH AUTOMATION SCRIPT
# ============================================================================
# Este script automatiza el proceso de push de datos, features y modelos a
# remote storage DVC. Incluye validaciones, logging y manejo de errores.
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

log_info "Iniciando proceso de DVC Push..."

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
dvc status

read -p "¿Deseas continuar con el push? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_warning "Push cancelado por el usuario"
    exit 0
fi

# ============================================================================
# PUSH DE DATOS Y MODELOS
# ============================================================================

log_info "Ejecutando DVC push..."

# Opción 1: Push completo (todos los archivos trackeados)
if [ "$1" == "--all" ] || [ -z "$1" ]; then
    log_info "Modo: Push completo de todos los artefactos"
    dvc push

# Opción 2: Push solo datos
elif [ "$1" == "--data" ]; then
    log_info "Modo: Push solo datos"
    dvc push data/01_raw.dvc data/02_intermediate.dvc data/03_primary.dvc data/04_feature.dvc

# Opción 3: Push solo modelos
elif [ "$1" == "--models" ]; then
    log_info "Modo: Push solo modelos"
    dvc push data/06_models.dvc

# Opción 4: Push específico
else
    log_info "Modo: Push específico de $1"
    dvc push "$1"
fi

if [ $? -eq 0 ]; then
    log_success "DVC push completado exitosamente"
else
    log_error "DVC push falló"
    exit 1
fi

# ============================================================================
# ACTUALIZAR GIT CON CAMBIOS DE DVC
# ============================================================================

log_info "Verificando cambios en archivos .dvc y dvc.lock..."

# Verificar si hay cambios en archivos DVC
if git status --porcelain | grep -E '\.dvc$|dvc\.lock$|dvc\.yaml$' > /dev/null; then
    log_warning "Hay cambios en archivos DVC que deben ser commiteados:"
    git status --short | grep -E '\.dvc$|dvc\.lock$|dvc\.yaml$'

    read -p "¿Deseas commitear estos cambios automáticamente? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add *.dvc dvc.lock dvc.yaml 2>/dev/null || true
        git commit -m "chore: Update DVC tracked files and metadata

- DVC push completed successfully
- Updated .dvc files and dvc.lock
- Timestamp: $(date '+%Y-%m-%d %H:%M:%S')

🤖 Generated with DVC automation script"

        log_success "Cambios commiteados exitosamente"
        log_info "No olvides hacer git push para sincronizar con el repositorio remoto"
    else
        log_warning "Recuerda commitear manualmente los cambios en archivos .dvc y dvc.lock"
    fi
else
    log_info "No hay cambios en archivos DVC para commitear"
fi

# ============================================================================
# RESUMEN
# ============================================================================

echo ""
log_success "=========================================="
log_success "DVC Push completado exitosamente"
log_success "=========================================="
log_info "Remote usado: $REMOTE"
log_info "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
log_info "Próximos pasos:"
log_info "  1. git push (si commiteaste cambios)"
log_info "  2. Verifica en tu remote storage que los archivos se subieron"
log_info "  3. Los colaboradores pueden hacer: dvc pull"
echo ""

# ============================================================================
# USO
# ============================================================================
# ./scripts/dvc_push.sh             # Push completo
# ./scripts/dvc_push.sh --all       # Push completo (explícito)
# ./scripts/dvc_push.sh --data      # Push solo datos
# ./scripts/dvc_push.sh --models    # Push solo modelos
# ./scripts/dvc_push.sh data/06_models.dvc  # Push específico
# ============================================================================
