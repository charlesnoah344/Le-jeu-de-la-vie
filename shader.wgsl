// Déclarations des ressources

// petite structure pour le stockage des dimensions
@group(0) @binding(0) var<uniform> size: vec2<u32>;
// stockage de la grille avant génération (lecture)
@group(0) @binding(1) var<storage, read> grid_in: array<u32>;
// stockage de la grille après génération (lecture + écriture)
@group(0) @binding(2) var<storage, read_write> grid_out: array<u32>;

// Détermine si la cellule est vivante ou morte en lisant sa valeur dans le grille de départ 
//préalablement transformée pour un comptage à plat (plus simple)
fn cell(x: i32, y: i32) -> u32 {
    let w = i32(size.x);
    let h = i32(size.y);
    return grid_in[u32(((y + h) % h) * w + (x + w) % w)]; // bords toriques (on empeche l'overflow dans la lecture des voisines)
}

// Point d'entrée du programme
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // On s'évite d'avoir des threads pour les positions hors de la grille
    if (id.x >= size.x || id.y >= size.y) { return; }
    // Passage en int signé pour les calculs
    let x = i32(id.x);
    let y = i32(id.y);
    var n = 0u; // Compteur de voisines vivantes.
    // On parcours les 8 voisines de la position actuelle (3*3)
    for (var dy = -1; dy <= 1; dy++) { 
        for (var dx = -1; dx <= 1; dx++) {
            // On exclus la position actuelle (je ne suis pas ma propre voisine) 
            // et compte mes voisines dans grace à cell qui ajoute 1 si j'ai 1 voisine vivante
            if (dx != 0 || dy != 0) { n += cell(x + dx, y + dy); }
        }
    }
    // Position à plat de i
    let i = id.y * size.x + id.x;
    // Application des règles du jeu de la vie
    grid_out[i] = select(0u, 1u, n == 3u || (grid_in[i] == 1u && n == 2u));
}