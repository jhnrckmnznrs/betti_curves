use std::collections::HashMap;

pub(crate) const INACTIVE_ROOT: u32 = u32::MAX;

#[derive(Debug)]
pub(crate) struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    pub(crate) fn new(n: usize) -> Self {
        assert!(
            n <= u32::MAX as usize,
            "UnionFind uses u32 indices; block has too many voxels"
        );

        let parent = (0..n as u32).collect();
        let rank = vec![0u8; n];

        Self { parent, rank }
    }

    pub(crate) fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            let gp = self.parent[p as usize];
            self.parent[x as usize] = gp;
            x = p;
        }
        x
    }

    pub(crate) fn union(&mut self, a: u32, b: u32) -> bool {
        let mut ra = self.find(a);
        let mut rb = self.find(b);

        if ra == rb {
            return false;
        }

        let rank_a = self.rank[ra as usize];
        let rank_b = self.rank[rb as usize];

        if rank_a < rank_b {
            std::mem::swap(&mut ra, &mut rb);
        }

        self.parent[rb as usize] = ra;

        if rank_a == rank_b {
            self.rank[ra as usize] += 1;
        }

        true
    }
}

#[derive(Debug)]
pub(crate) struct FaceLabels {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) roots: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LocalRootKey {
    pub(crate) slab_id: usize,
    pub(crate) root: u32,
}

#[derive(Debug)]
pub(crate) struct DynamicUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl DynamicUnionFind {
    pub(crate) fn new() -> Self {
        Self {
            parent: Vec::new(),
            rank: Vec::new(),
        }
    }

    pub(crate) fn make_set(&mut self) -> u32 {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        self.rank.push(0);
        id
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            let gp = self.parent[p as usize];
            self.parent[x as usize] = gp;
            x = p;
        }
        x
    }

    pub(crate) fn union(&mut self, a: u32, b: u32) -> bool {
        let mut ra = self.find(a);
        let mut rb = self.find(b);

        if ra == rb {
            return false;
        }

        let rank_a = self.rank[ra as usize];
        let rank_b = self.rank[rb as usize];

        if rank_a < rank_b {
            std::mem::swap(&mut ra, &mut rb);
        }

        self.parent[rb as usize] = ra;

        if rank_a == rank_b {
            self.rank[ra as usize] += 1;
        }

        true
    }

    pub(crate) fn len(&self) -> usize {
        self.parent.len()
    }
}

pub(crate) fn get_or_create_global_id(
    key: LocalRootKey,
    id_map: &mut HashMap<LocalRootKey, u32>,
    uf: &mut DynamicUnionFind,
) -> u32 {
    if let Some(&id) = id_map.get(&key) {
        id
    } else {
        let id = uf.make_set();
        id_map.insert(key, id);
        id
    }
}

#[derive(Debug)]
pub(crate) struct DynamicOutsideUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    touches_outside: Vec<bool>,
}

impl DynamicOutsideUnionFind {
    pub(crate) fn new() -> Self {
        Self {
            parent: Vec::new(),
            rank: Vec::new(),
            touches_outside: Vec::new(),
        }
    }

    pub(crate) fn make_set(&mut self, touches_outside: bool) -> u32 {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        self.rank.push(0);
        self.touches_outside.push(touches_outside);
        id
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            let gp = self.parent[p as usize];
            self.parent[x as usize] = gp;
            x = p;
        }
        x
    }

    pub(crate) fn union(&mut self, a: u32, b: u32) -> Option<(bool, bool)> {
        let mut ra = self.find(a);
        let mut rb = self.find(b);

        if ra == rb {
            return None;
        }

        let a_outside = self.touches_outside[ra as usize];
        let b_outside = self.touches_outside[rb as usize];

        let rank_a = self.rank[ra as usize];
        let rank_b = self.rank[rb as usize];

        if rank_a < rank_b {
            std::mem::swap(&mut ra, &mut rb);
        }

        self.parent[rb as usize] = ra;

        if rank_a == rank_b {
            self.rank[ra as usize] += 1;
        }

        self.touches_outside[ra as usize] = a_outside || b_outside;

        Some((a_outside, b_outside))
    }

    pub(crate) fn len(&self) -> usize {
        self.parent.len()
    }
}

pub(crate) fn get_or_create_global_outside_id(
    key: LocalRootKey,
    touches_outside: bool,
    id_map: &mut HashMap<LocalRootKey, u32>,
    uf: &mut DynamicOutsideUnionFind,
) -> u32 {
    if let Some(&id) = id_map.get(&key) {
        id
    } else {
        let id = uf.make_set(touches_outside);
        id_map.insert(key, id);
        id
    }
}
