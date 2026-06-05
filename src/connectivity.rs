use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy)]
pub enum Connectivity {
    Six,
    TwentySix,
}

pub fn parse_connectivity(s: &str) -> Result<Connectivity> {
    match s {
        "6" | "six" | "Six" => Ok(Connectivity::Six),
        "26" | "twenty-six" | "TwentySix" | "twentysix" => Ok(Connectivity::TwentySix),
        _ => bail!("Connectivity must be either 6 or 26, got {:?}", s),
    }
}

pub fn dual_connectivity(foreground: Connectivity) -> Connectivity {
    match foreground {
        Connectivity::Six => Connectivity::TwentySix,
        Connectivity::TwentySix => Connectivity::Six,
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Mode {
    Betti0,
    Betti2,
    Both,
    EventBetti0,
    EventBetti2,
}

pub fn parse_mode(s: &str) -> Result<Mode> {
    match s {
        "betti0" | "b0" | "0" => Ok(Mode::Betti0),
        "betti2" | "b2" | "2" => Ok(Mode::Betti2),
        "both" | "all" => Ok(Mode::Both),
        "event-betti0" | "event_betti0" | "eb0" => Ok(Mode::EventBetti0),
        "event-betti2" | "event_betti2" | "eb2" => Ok(Mode::EventBetti2),
        _ => bail!(
            "Mode must be betti0, betti2, both, event-betti0, or event-betti2, got {:?}",
            s
        ),
    }
}
