use std::io::{self, Read};

/// Read one complete fixed-size field.
///
/// Returns `Ok(false)` only when EOF occurs before any byte of the field. A
/// partially present field is corruption and therefore returns `UnexpectedEof`.
pub(crate) fn read_exact_or_eof(reader: &mut impl Read, buffer: &mut [u8]) -> io::Result<bool> {
    let mut filled = 0usize;

    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "truncated binary record: read {filled} of {} field bytes",
                        buffer.len()
                    ),
                ));
            }
            Ok(count) => filled += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn distinguishes_clean_eof_from_a_partial_field() {
        let mut field = [0u8; 2];
        assert!(!read_exact_or_eof(&mut Cursor::new([]), &mut field).unwrap());

        let error = read_exact_or_eof(&mut Cursor::new([7]), &mut field).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);

        assert!(read_exact_or_eof(&mut Cursor::new([7, 9]), &mut field).unwrap());
        assert_eq!(field, [7, 9]);
    }
}
