import numpy as np
from flygarden.vision_tracks import MaskedExpansion


def bundle(radius=0,shift=0,body_radius=0):
    image=np.full((2,100,100,3),255,dtype=np.uint8);mask=np.ones((2,100,100),dtype=bool)
    yy,xx=np.indices((100,100))
    if radius:image[0][(xx-60-shift)**2+(yy-40)**2<=radius**2]=0
    if body_radius:
        self_pixel=(xx-20)**2+(yy-75)**2<=body_radius**2
        image[0][self_pixel]=0;mask[0][self_pixel]=False
    return {'rgb':image,'valid':mask}


def test_masks_self_expansion_but_tracks_smaller_external_object():
    encoder=MaskedExpansion(bundle(body_radius=15));rates=[]
    for radius,body_radius in zip((3,5,7,9),(18,21,24,27)):
        rates.append(encoder.advance(bundle(radius=radius,body_radius=body_radius),1/30)[0]['lplc2_hz'])
    assert rates[:2]==[0,0] and rates[2]>0 and rates[3]>0
    empty=MaskedExpansion(bundle(body_radius=15))
    for radius in (18,21,24,27):assert empty.advance(bundle(body_radius=radius),1/30)[0]['lplc2_hz']==0


def test_translation_remains_quiet_and_restore_tracks_exactly():
    encoder=MaskedExpansion(bundle())
    for shift in (0,1,2,3):assert encoder.advance(bundle(radius=8,shift=shift),1/30)[0]['lplc2_hz']==0
    clone=MaskedExpansion(bundle());clone.restore(encoder.snapshot())
    assert encoder.advance(bundle(radius=10,shift=3),1/30)==clone.advance(bundle(radius=10,shift=3),1/30)
